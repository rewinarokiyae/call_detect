# Scam Detection Pipeline (Spam Model)

This module implements a multi-modal scam detection system using audio and text features, optimized for low-VRAM environments (RTX 3050 4GB).

## Structure

- **`data_spam/`**: Dataset and feature storage.
  - `metadata.csv`: Training metadata.
  - `splits/`: Train/Val/Test split CSVs.
  - `features/`: Pre-computed feature tensors (.pt).
- **`audio_loader/`**: Handles loading of agent and customer audio files.
- **`asr/`**: Automatic Speech Recognition using Whisper (Base model).
- **`audio_features/`**: Audio embedding extraction using SpeechBrain (ECAPA-TDNN).
- **`text_features/`**: Text feature extraction (sensitive keywords, urgency, Groq placeholder).
- **`fusion/`**: Combines audio and text features.
- **`model/`**: PyTorch classifier model and scripts.
  - `train.py`: Training loop with Focal Loss and Early Stopping.
  - `evaluate.py`: Metrics generation (Accuracy, F1, FPR).
  - `classifier.py`: MLP Architecture.
- **`inference/`**: Main inference script (`predict.py`).
- **`utils/`**: Utility functions (audio processing).

## workflow

### 1. Data Preparation

Ensure `spam_model/data_spam/metadata.csv` exists with columns:
`file_id`, `agent_audio_path`, `customer_audio_path`, `label`, `transcript_agent`, `transcript_customer`.

Run the preparation script to create splits:
```bash
python spam_model/data_spam/prepare_data.py
```

### 2. Feature Pre-computation (Critical for 4GB VRAM)

To avoid OOM during training, pre-compute features:
```bash
python spam_model/data_spam/precompute_features.py
```

### 3. Training

Train the model using the pre-computed features:
```bash
python spam_model/model/train.py
```
This saves checkpoints to `spam_model/checkpoints/`.

### 4. Evaluation

Generate a performance report on the test set:
```bash
python spam_model/model/evaluate.py
```
View the report at: `spam_model/evaluation/report.txt`.

### 5. Inference

Run the prediction script with audio paths:
```bash
python spam_model/inference/predict.py --agent path/to/agent.wav --customer path/to/customer.wav
```
Configurable threshold in `config.yaml`.

## Hardware Constraints

- **VRAM**: 4GB Limit. Models are loaded sequentially.
- **FFMPEG**: Not required. Uses `soundfile`.
- **Latency**: ASR + ECAPA runs in ~5-10s on GPU.

## Configuration

All paths and hyperparameters (thresholds, epochs, LR) are defined in `spam_model/config.yaml`.
