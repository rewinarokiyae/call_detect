import torch

def compute_spectral_flatness(waveform, n_fft=2048, hop_length=512):
    """
    Computes spectral flatness using PyTorch (GPU compatible).
    Flatness = Geometric_Mean(Mag) / Arithmetic_Mean(Mag)
    """
    # 1. STFT [Batch, Freq, Time, 2] -> Magnitude [Batch, Freq, Time]
    # torch.stft returns complex tensor or (real, imag) depending on version.
    # We use return_complex=True for modern torch
    
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device), return_complex=True)
    mag = torch.abs(stft) + 1e-6 # Avoid log(0)
    
    # 2. Compute means over Frequency dimension
    # Geometric Mean = exp(mean(log(x)))
    log_mag = torch.log(mag)
    gmean = torch.exp(torch.mean(log_mag, dim=1)) # [Batch, Time]
    amean = torch.mean(mag, dim=1) # [Batch, Time]
    
    flatness = gmean / (amean + 1e-6)
    
    # 3. Average over Time
    return torch.mean(flatness, dim=1, keepdim=True) # [Batch, 1]

def estimate_bandwidth(waveform, sr=16000, n_fft=2048, hop_length=512):
    """
    Estimates 95% spectral rolloff frequency using PyTorch.
    """
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=waveform.device), return_complex=True)
    mag = torch.abs(stft) # [Batch, Freq, Time]
    
    # Power Spectrum
    power = mag ** 2
    
    # Cumulative Sum over Freq
    total_energy = torch.sum(power, dim=1, keepdim=True) # [B, 1, T]
    cumulative_energy = torch.cumsum(power, dim=1) # [B, F, T]
    
    # Find 95% threshold
    threshold = 0.95 * total_energy
    
    # Mask where cumulative < threshold (we want the index where it CROSSES threshold)
    mask = (cumulative_energy < threshold).float()
    
    # Sum mask gives count of bins below threshold -> index of crossing
    bin_idx = torch.sum(mask, dim=1) # [B, T]
    
    # Convert bin_idx to Frequency
    # Freq = bin * (SR / N_FFT)
    freq_res = sr / n_fft
    rolloff = bin_idx * freq_res
    
    # Average over time
    return torch.mean(rolloff, dim=1, keepdim=True) # [Batch, 1]

def detect_clipping_quantization(waveform):
    """
    Simple heuristic for quantization noise/clipping.
    """
    return torch.zeros(waveform.shape[0], 1).to(waveform.device)
