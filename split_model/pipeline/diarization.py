import numpy as np
import librosa
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import soundfile as sf
import os
import torch
import torchaudio

# Removed top-level patch and import
import parselmouth
# from speechbrain.inference.speaker import EncoderClassifier # MOVED TO __INIT__

class ProductionDiarizer:
    def __init__(self, n_clusters=2, window_len=0.5, hop_len=0.1, sr=16000):
        self.n_clusters = n_clusters
        self.window_len = window_len
        self.hop_len = hop_len
        self.sr = sr
        
        # Load Embedding Model using SpeechBrain
        print("Loading ECAPA-TDNN embedding model...")
        
        # --- LOCAL PATCH ---
        import sys
        import types
        import torchaudio
        if not hasattr(torchaudio, 'backend'):
            torchaudio.backend = types.ModuleType("backend")
            sys.modules["torchaudio.backend"] = torchaudio.backend
        if not hasattr(torchaudio.backend, "list_audio_backends"):
            torchaudio.backend.list_audio_backends = lambda: []
        # -------------------

        from speechbrain.inference.speaker import EncoderClassifier
        
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            # savedir="pretrained_models/spkrec-ecapa-voxceleb", # Removed to avoid WinError 1314 (Symlink)
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        print("[OK] SpeechBrain Embedding Model Loaded Successfully")

    def extract_voice_quality(self, y, sr):
        """
        Extract Jitter, Shimmer, HNR using Parselmouth (Praat).
        """
        try:
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            pitch = sound.to_pitch()
            pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
            
            # Jitter
            jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            
            # Shimmer
            shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # HNR
            harmonicity = sound.to_harmonicity()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            
            return [jitter, shimmer, hnr]
        except:
            return [0.0, 0.0, 0.0]

    def extract_features(self, y, sr):
        """
        Extract comprehensive feature set:
        A. Acoustic (MFCC, Spectral)
        B. Prosodic (Pitch, Energy)
        C. Voice Quality (Jitter, Shimmer)
        D. Embeddings (ECAPA-TDNN)
        """
        # --- A. Acoustic & D. Spectral ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flux = librosa.onset.onset_strength(y=y, sr=sr) # Proxy for calc
        zcr = librosa.feature.zero_crossing_rate(y)
        
        acoustic_feats = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            [np.mean(centroid), np.std(centroid)],
            [np.mean(rolloff), np.std(rolloff)],
            [np.mean(flux), np.std(flux)],
            [np.mean(zcr), np.std(zcr)]
        ])
        
        # --- B. Prosodic ---
        rms = librosa.feature.rms(y=y)
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=2048)
        f0 = np.nan_to_num(f0)
        
        prosodic_feats = np.concatenate([
            [np.mean(rms), np.std(rms)],
            [np.mean(f0), np.std(f0)]
        ])
        
        # --- C. Voice Quality ---
        vq_feats = np.array(self.extract_voice_quality(y, sr))
        
        # --- E. Embeddings ---
        embedding_feat = np.zeros(192) # default size for ECAPA
        if self.embedding_model:
            # EncoderClassifier expects tensor [batch, time]
            tensor_y = torch.tensor(y).unsqueeze(0)
            with torch.no_grad():
                emb = self.embedding_model.encode_batch(tensor_y)
                # emb is [batch, 1, 192] -> squeeze to [192]
                embedding_feat = emb.squeeze().cpu().numpy()

        # Concatenate ALL
        # We might need to weigh them, but for now we concat and let StandardScaler handle scale
        full_features = np.concatenate([
            acoustic_feats,
            prosodic_feats,
            vq_feats,
            embedding_feat
        ])
        
        return full_features

    def process(self, file_path):
        """
        Full Pipeline: VAD -> Windowing -> Feature Extraction -> Clustering
        """
        print(f"Processing {file_path} with PRODUCTION metrics...")
        y, sr = librosa.load(file_path, sr=self.sr)
        
        # 1. Simple VAD / Trimming (per requirements)
        # We iterate windows and drop silence later or skip classification
        
        window_samples = int(self.window_len * sr)
        hop_samples = int(self.hop_len * sr)
        
        feature_list = []
        timestamps = []
        
        print("Extracting features (This may take a while)...")
        for start in range(0, len(y) - window_samples, hop_samples):
            end = start + window_samples
            chunk = y[start:end]
            
            # Quick energy VAD
            if np.mean(chunk**2) < 0.0001: # Silence threshold
                continue
                
            feat = self.extract_features(chunk, sr)
            if np.isnan(feat).any():
                feat = np.nan_to_num(feat)
                
            feature_list.append(feat)
            timestamps.append((start / sr, end / sr))
            
        if not feature_list:
            print("No speech detected.")
            return [], y, sr
            
        X = np.array(feature_list)
        print(f"Extracted feature matrix: {X.shape}")
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality Reduction (Optional but good for embedding fusion)
        # PCA to reduce noise from so many features (192 embeddings + ~60 others)
        pca = PCA(n_components=min(50, X_scaled.shape[0], X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering
        print("Clustering...")
        n_clusters = self.n_clusters
        # Spectral Clustering
        clusterer = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=min(10, X_pca.shape[0]-1),
            random_state=42,
            n_jobs=-1
        )
        labels = clusterer.fit_predict(X_pca)
        
        # Merge Segments
        segments = []
        if len(labels) == 0:
            return segments, y, sr
            
        current_speaker = labels[0]
        start_time = timestamps[0][0]
        
        for i in range(1, len(labels)):
            # Check for gap in timestamps (if VAD skipped windows)
            time_gap = timestamps[i][0] - timestamps[i-1][1]
            
            if labels[i] != current_speaker or time_gap > 0.1:
                end_time = timestamps[i-1][1]
                segments.append({
                    'start': start_time, 
                    'end': end_time, 
                    'speaker': int(current_speaker)
                })
                current_speaker = labels[i]
                start_time = timestamps[i][0]
                
        segments.append({
            'start': start_time, 
            'end': timestamps[-1][1], 
            'speaker': int(current_speaker)
        })
        
        # Post-Processing: Minimum Duration & Smoothing (Agglomerate short segments)
        # TODO: Implement simple smoothing here if needed
        
        return segments, y, sr

    def save_segments(self, segments, y, sr, output_dir="output_segments"):
        # Existing logic, ensure PCM_16
        os.makedirs(output_dir, exist_ok=True)
        speaker_audio = {i: [] for i in range(self.n_clusters)}
        
        for seg in segments:
            start = int(seg['start'] * sr)
            end = int(seg['end'] * sr)
            speaker_audio[seg['speaker']].append(y[start:end])
            
        paths = {}
        for spk_id, chunks in speaker_audio.items():
            if not chunks: continue
            combined = np.concatenate(chunks)
            out_path = os.path.join(output_dir, f"speaker_{spk_id}.wav")
            sf.write(out_path, combined, sr, subtype='PCM_16')
            paths[spk_id] = out_path
            
        return paths

# Alias for compatibility if needed, but we should update imports
Diarizer = ProductionDiarizer
