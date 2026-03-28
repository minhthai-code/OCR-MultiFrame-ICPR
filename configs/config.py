"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    MODEL_TYPE: str = "restran"  # "crnn" or "restran"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network
    
    # Data paths
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"
    
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #    # Training hyperparameters
    """
    BATCH_SIZE (64)
    Number of samples processed before updating model weights.
    Example: If your dataset has 6400 images and batch size is 64 → 100 steps per epoch.
    Larger batch → more stable gradients but uses more GPU memory.
    Smaller batch → more updates per epoch, noisier gradient but less memory.
    LEARNING_RATE (5e-4)
    Determines how big the weight updates are during training.
    Too high → model may diverge, too low → slow training.
    Example: 5e-4 = 0.0005 (a common starting point for Adam optimizer).
    EPOCHS (30)
    Number of times the entire training dataset is passed through the model.
    SEED (42)
    Fixes random number generators to reproduce results.
    NUM_WORKERS (10)
    Number of parallel CPU threads used to load data.
    More workers → faster data loading, but uses more CPU.
    Too many workers → can slow down due to context switching.
    WEIGHT_DECAY (1e-4)
    Regularization parameter to prevent overfitting.
    Adds a small penalty to large weights (like L2 regularization).
    GRAD_CLIP (5.0)
    Gradient clipping to prevent exploding gradients.
    Any gradient larger than 5 is scaled down.
    SPLIT_RATIO (0.9)
    Fraction of the training dataset used as actual training (0.9 = 90%), rest for validation.
    USE_CUDNN_BENCHMARK
    When True, cuDNN will optimize GPU kernels for your model/input sizes. Can make training faster but non-deterministic.
    """
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False
    
    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    
    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"
    
    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()
