"""Main entry point for OCR training pipeline."""

"""
    It orchestrates the entire training pipeline, including:
        CLI arguments
        ↓
        Load Config
            ↓
        Create Dataset
            ↓
        Create DataLoader
            ↓
        Build Model
            ↓
        Train Model
            ↓
        (Optional) Generate Submission
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    # Create a descriptive help message for the CLI tool and collect all allowed command-line arguments.
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )

    # Define CLI arguments that can override config values. Each argument has a default of None, it will fall back to the value defined in the Config class if not provided.
    
    # python train.py --experiment-name my_exp --model restran --epochs 50 --batch-size 64 --lr 0.001 --data-root /path/to/data --seed 1234 --num-workers 4 --hidden-size 256 --transformer-heads 8 --transformer-layers 4 --aug-level full --output-dir results/ --no-stn --submission-mode
    # Or: python train.py \
    # --model restran \
    # --experiment-name my_experiment \
    # --data-root /path/to/dataset \
    # --batch-size 64 \
    # --epochs 30 \
    # --lr 0.0005 \
    # --aug-level full

    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran"], default=None,
        help="Model architecture: 'crnn' or 'restran' (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )
    parser.add_argument(
        "--aug-level", type=str, choices=["full", "light"], default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save checkpoints and submission files (default: results/)",
    )
    parser.add_argument(
        "--no-stn", action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--submission-mode", action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""

    # Parse CLI arguments from command line and override config values
    args = parse_args()
    
    # Initialize base configuration, CLI arguments may override default values defined in the Config
    config = Config()
    
    """
        The CLI arguments are designed to override the default values defined in the Config dataclass.
        Python dictionary. Structure: key : value
        key: name of the CLI argument (e.g., 'experiment_name', 'model', 'epochs')
        value: corresponding attribute name in the Config class (e.g., 'EXPERIMENT_NAME', 'MODEL_TYPE', 'EPOCHS')
        only ask these keys from the arugemnts else use the default value from the config class
    """
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
    }
    
    """
        .items() in Python returns: key-value pairs of the dictionary as tuples.
        like ('experiment_name', 'EXPERIMENT_NAME'), ('model', 'MODEL_TYPE'), etc.
        Python tuple unpacking in the loop we assign first value for example 'experiment_name' to variable arg_name and second value 'EXPERIMENT_NAME' to variable config_name.
        getattr(args, arg_name, None) so if the args.experiment_name is provided in the command line, it will return that value, otherwise it will return None.
        If the value is not None, we use setattr(config, config_name, value) to set the object to the value provided via CLI. 
        For example, if --experiment-name my_exp is provided, it will set config.EXPERIMENT_NAME = "my_exp". -> override the config value with the CLI argument value 
     """
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None) # Get the value of the CLI argument (e.g., args.experiment_name)
        if value is not None:
            setattr(config, config_name, value) # Override the config attribute with the CLI argument value (e.g., config.EXPERIMENT_NAME = "my_exp")
    
    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level # Override augmentation level if provided via CLI
    
    if args.no_stn:
        config.USE_STN = False # Disable STN if --no-stn flag is set in CLI
    
    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    """
    Purpose:

    This function ensures your machine learning experiments are reproducible, meaning you get the same results every time you run your code.

    It sets fixed random seeds for:

    Python (random)
    NumPy (np)
    PyTorch (CPU and GPU)

    This makes operations like:

    weight initialization
    data shuffling
    random sampling

    behave consistently across runs.

    GPU Behavior (cuDNN Settings)

    The function also controls how CUDA (GPU acceleration) behaves:
    Deterministic mode (benchmark=False)
    → Slower but produces exact same results every run
    Benchmark mode (benchmark=True)
    → Faster but may produce slightly different results each run
    
    Tradeoff:
    Reproducibility → good for debugging, research
    Speed → good for training and performance

    You choose depending on your goal.

    Overall Benefit
    By removing randomness and controlling GPU behavior, this function makes:
    debugging easier
    experiments fairer
    results more reliable
    """
    seed_everything(config.SEED)
    
    # print the final configuration after applying CLI overrides
    print("\n Configuration ".center(60, "="))
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters that are shared by multiple dataset objects, use all from config no CLI overrides needed here
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
    }
    
    """
        Create datasets based on mode for later to create dataloaders.
        1. Normal training mode -> split dataset into train + validation
        2. Submission mode -> train on FULL dataset and generate predictions for test data
    """
    if args.submission_mode:
        # In competitions, once the model is tuned using validation,
        # we retrain on the FULL dataset to maximize training data before generating final predictions.
        print("\n SUBMISSION MODE ENABLED ".center(60, "="))
        print("   - Training on FULL dataset (no validation split)")
        print("   - Generate predictions for test data after training\n")
        
        # Create training dataset with full_train=True
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True, # full_train=True tells the dataset class to skip the train/val split
            **common_ds_params
        )
        
        # Create test dataset if test data exists contain images only (no labels)
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            # The test dataset is created in 'val' mode to apply the same resizing and normalization transforms, but with is_test=True to indicate that there are no labels and we want to return image paths for submission.
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            # DataLoader converts dataset into batches for efficient GPU processing
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print(f"WARNING: Test data not found at {config.TEST_DATA_ROOT}")
        
        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train', # training mode applies data augmentation
            **common_ds_params
        )
        
        # Validation dataset use to evaluate model performance during training, it uses the remaining portion of data (no augmentation, just resize + normalize)
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )
        
        # Create DataLoader only if validation set is not empty
        val_loader = None
        if len(val_ds) > 0:
            # Validation loader processes validation data in batches
            # Used to compute validation loss/accuracy after each epoch
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print("WARNING: Validation dataset is empty.")
        
        test_loader = None
    
    # Safety check to prevent training with empty dataset
    if len(train_ds) == 0:
        print("Training dataset is empty!")
        sys.exit(1)

    # Create training data loader, DataLoader handles batching, shuffling, multiprocessing, and memory transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model based on config.MODEL_TYPE, either ResTranOCR or MultiFrameCRNN. The model is moved to the device specified in config (GPU if available, otherwise CPU).
    if config.MODEL_TYPE == "restran":

        # Transformer-based OCR model
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
    else:

        # CRNN model (CNN + recurrent layers)
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
    
    # Print model summary, numel() counts the number of elements (parameters) in each tensor, we sum them up to get total parameters and trainable parameters. This gives us an idea of the model size and complexity.
    total_params = sum(p.numel() for p in model.parameters())

    # Only parameters with requires_grad=True will be updated during training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    # Initialize trainer and start training
    # trainer handles: training loop, 
    # forward pass
    # loss computation
    # backpropagation
    # optimizer step
    # validation
    # checkpoint saving, etc.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    # Start training loop
    trainer.fit()
    
    # Run test inference in submission mode
    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("GENERATING SUBMISSION FILE")
        print("="*60)
        
        # Load best checkpoint if it exists saves the best performing model on validation set
        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")

        if os.path.exists(best_model_path):
            print(f"Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE)) # load_state_dict loads saved model weights 
        else:
            print("No best checkpoint found, using final model weights")
        
        # Run inference on test data and generate submission file
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()
