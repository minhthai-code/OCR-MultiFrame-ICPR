# OCR-MultiFrame-ICPR: OCR For Multi-Frame License Plate Recognition

**A state-of-the-art deep learning solution for low-resolution license plate recognition in the ICPR 2026 Challenge.**

This repository presents a comprehensive implementation for robust optical character recognition (OCR) on low-resolution license plates. The system leverages temporal information from video frame sequences combined with advanced attention fusion mechanisms and transformer-based architectures to achieve superior recognition accuracy.

**Challenge:** [ICPR 2026 Low-Resolution License Plate Recognition](https://icpr26lrlpr.github.io/)

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- CUDA-enabled GPU (recommended for training)

### Installation
```bash
git clone https://github.com/minhthai-code/OCR-MultiFrame-ICPR
cd OCR-MultiFrame-ICPR
```

Using `pip` (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

### Basic Usage

```bash
# Train with default settings (ResTranOCR + STN)
python train.py

# Train CRNN baseline
python train.py --model crnn --experiment-name crnn_baseline

# Generate submission file
python train.py --submission-mode --model restran
```

## Overview

### Key Features

- **Multi-Frame Fusion**: Processes 5-frame sequences with attention-based temporal fusion
- **Spatial Transformer Network**: Optional STN module for automatic geometric normalization
- **Dual Architectures**: CRNN (baseline) and ResTranOCR (ResNet34 + Transformer encoder)
- **Advanced Data Augmentation**: Scenario-aware validation split with configurable augmentation strategies
- **Production-Ready Training**: Mixed precision training, gradient clipping, and OneCycleLR scheduling
- **Comprehensive Evaluation**: Integrated ablation studies and submission file generation

## Model Architectures

### CRNN (Convolutional Recurrent Neural Network)
**Pipeline:** Multi-frame Input → STN Alignment → CNN Feature Extraction → Attention Fusion → BiLSTM Sequence Modeling → CTC Decoding

A lightweight baseline model combining convolutional feature extraction with bidirectional LSTM for sequence-to-sequence learning. Suitable for resource-constrained environments while maintaining competitive accuracy.

### ResTranOCR (ResNet Transformer OCR)
**Pipeline:** Multi-frame Input → STN Alignment → ResNet34 Backbone → Attention Fusion → Transformer Encoder → CTC Decoding

A modern, high-capacity architecture leveraging a ResNet34 backbone for feature extraction and a Transformer encoder with positional encoding for improved modeling of long-range character dependencies.

**Input/Output Specification:**
- **Input shape:** $(B, 5, 3, 32, 128)$ where $B$ is batch size, 5 is number of frames, 3 is RGB channels, and 32×128 is the normalized plate image size
- **Output:** Character sequences via CTC decoding

---

## Usage

### Dataset Preparation

Organize your dataset according to the following structure:

```
data/train/
├── track_001/
│   ├── lr-001.png
│   ├── lr-002.png
│   ├── ...
│   ├── hr-001.png (optional, for synthetic low-resolution generation)
│   └── annotations.json
└── track_002/
    └── ...
```

The `annotations.json` file should contain the ground-truth plate text:
```json
{"plate_text": "ABC1234"}
```

### Training Models

#### Basic Training

```bash
python train.py
```

#### Advanced Configuration

```bash
python train.py \
    --model restran \
    --experiment-name my_experiment \
    --data-root /path/to/dataset \
    --batch-size 64 \
    --epochs 30 \
    --lr 0.0005 \
    --aug-level full
```

#### Disable Spatial Transformer Network

```bash
python train.py --no-stn
```

#### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | `restran` | Model type: `crnn` or `restran` |
| `--experiment-name` | `-n` | None | Experiment identifier for tracking |
| `--data-root` | N/A | `data/train` | Path to training dataset |
| `--batch-size` | N/A | `64` | Training batch size |
| `--epochs` | N/A | `30` | Total training epochs |
| `--lr` | N/A | `5e-4` | Initial learning rate |
| `--aug-level` | N/A | `full` | Augmentation level: `full` or `light` |
| `--no-stn` | N/A | False | Disable Spatial Transformer Network module |
| `--submission-mode` | N/A | False | Train on full dataset and generate test predictions |
| `--output-dir` | N/A | `results/` | Output directory for checkpoints and results |

---

## Ablation Study

Execute comprehensive ablation experiments to evaluate the impact of different architectural choices:

```bash
python run_ablation.py
```

This script automatically runs the following configurations:
- CRNN with and without STN module
- ResTranOCR with and without STN module

Results are aggregated in `experiments/ablation_summary.txt` for easy comparison.

### Output Artifacts

After training completes, the following artifacts are generated in the output directory:

| File | Description |
|------|-------------|
| `{experiment_name}_best.pth` | Best model checkpoint (lowest validation loss) |
| `submission_{experiment_name}.txt` | Test predictions in competition format: `track_id,predicted_text;confidence` |

## Configuration

Hyperparameters are centrally managed in [configs/config.py](configs/config.py). Below are the primary configuration parameters:

```python
# Model Selection
MODEL_TYPE = "restran"           # "crnn" or "restran"
USE_STN = True                   # Enable/disable Spatial Transformer Network

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 30
AUGMENTATION_LEVEL = "full"      # "full" or "light"

# CRNN Configuration
HIDDEN_SIZE = 256
RNN_DROPOUT = 0.25

# ResTranOCR Configuration
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 2048
TRANSFORMER_DROPOUT = 0.1
```

All configuration parameters can be overridden via command-line arguments when invoking `train.py`.

## Project Structure

```
OCR-MultiFrame-ICPR/
├── configs/
│   ├── __init__.py
│   └── config.py                    # Configuration and hyperparameters
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # MultiFrameDataset with scenario-aware splitting
│   │   └── transforms.py            # Augmentation and preprocessing pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── components.py            # Shared modules (STN, AttentionFusion, etc.)
│   │   ├── crnn.py                  # CRNN baseline implementation
│   │   └── restran.py               # ResTranOCR advanced model
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py               # Training loop and validation logic
│   └── utils/
│       ├── __init__.py
│       ├── common.py                # Utility functions and helpers
│       └── postprocess.py           # CTC decoding and post-processing
├── experiments/                      # Ablation study results
├── train.py                          # Main training entry point
├── run_ablation.py                   # Ablation study automation script
├── README.md
└── pyproject.toml                    # Project dependencies and metadata
```

## Technical Details

### Attention Fusion Mechanism

The attention fusion module dynamically computes normalized attention weights across temporal frames, enabling the model to selectively emphasize informative frames while suppressing noise. Multi-frame features are fused into a unified representation before downstream sequence modeling stages.

### Data Augmentation Strategies

The system implements two augmentation profiles:

- **Full Augmentation**: Affine transformations, perspective warping, HSV color adjustment, coarse dropout, and elastic distortions
- **Light Augmentation**: Normalized resize and channel-wise normalization only

The validation set uses scenario-aware stratified splitting to prioritize challenging scenarios, preventing model overfitting to easier examples.

### Loss and Optimization

The training pipeline employs:
- **Loss Function**: Connectionist Temporal Classification (CTC) for handling variable-length sequences
- **Optimizer**: Adam with mixed precision training
- **Learning Rate Schedule**: OneCycleLR for efficient convergence
- **Regularization**: Gradient clipping and dropout to prevent overfitting

---

## License

This project is provided as-is for research and educational purposes in the ICPR 2026 Challenge. Refer to the LICENSE file for detailed terms.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{OCR-MultiFrame-ICPR-2026,
  title={OCR-MultiFrame-ICPR: Multi-Frame License Plate Recognition},
  author={Your Name},
  year={2026},
  url={https://github.com/minhthai-code/OCR-MultiFrame-ICPR}
}
```

## Authors

**Development Team**: Tran Minh Thai, Nguyen Huu Tin

## Acknowledgments

This implementation is developed for the ICPR 2026 Challenge on Low-Resolution License Plate Recognition. We acknowledge the challenge organizers and the broader computer vision community for their valuable contributions to this domain.