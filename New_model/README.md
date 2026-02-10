# ASVspoof5 Track-1 Countermeasure (New Model)

This is a consolidated, production-ready implementation of the ASVspoof5 Track-1 Countermeasure system.

## Architecture
- **Streams**: WavLM (Pre-trained), Log-Mel Spectrogram (+SE), Phase (MGD/SE).
- **Fusion**: Attention-based fusion of all streams.
- **Pooling**: Attentive Statistical Pooling (ASP).
- **Loss**: Focal Loss (Gamma=2.0).

## Directory Structure
- `config.yaml`: All hyperparameters.
- `model.py`: The Neural Network definition.
- `dataset.py`: Data loading and protocol parsing.
- `train.py`: Training entry point.
- `utils.py`: Loss functions and metrics.

## How to Run on Vast.ai

1. **Upload Dataset**: Ensure data is at `/workspace/data/asvspoof5` (or update `config.yaml`).
2. **Setup Environment**:
   ```bash
   bash setup_env.sh
   ```
3. **Start Training**:
   ```bash
   python train.py --config config.yaml
   ```

## Configuration
Edit `config.yaml` to enable/disable features or change training parameters.

## Resume Training
To resume from a checkpoint:
```bash
# Modify train.py to load state_dict if needed, 
# or simply start fresh as this script saves 'last_model.pth' and 'best_model.pth'.
```

## Features
- **Early Stopping**: Monitors Validation EER.
- **Mixed Precision**: automatically enabled if CUDA is present.
- **WavLM**: Downloads `microsoft/wavlm-large` automatically on first run.
