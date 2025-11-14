import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from tqdm import tqdm
import pandas as pd

from unet import UNet
from dataset import get_dataloader

if __name__ == "__main__":
    # Configuration
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    IMAGE_SIZE = 128
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    LOAD_MODEL_PATH = "checkpoints/my_model.pth"
    DATASET_PATH = "data/processed/training_data.csv"
    MODEL_NAME = "unet.pth"
    CHECKPOINT_DIR = "checkpoints"
    
    # Testing mode: Use subset of data for quick local testing
    TEST_MODE = True  # Set to False for full training
    TEST_LIMIT = 100  # Only use first 100 samples for testing
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load dataset
    if TEST_MODE:
        # For local testing: limit dataset size
        df = pd.read_csv(DATASET_PATH)
        test_df = df.head(TEST_LIMIT)
        test_csv = "data/processed/training_data_test.csv"
        test_df.to_csv(test_csv, index=False)
        print(f"TEST MODE: Using {TEST_LIMIT} samples for quick testing")
        dataset_path = test_csv
    else:
        dataset_path = DATASET_PATH
    
    train_loader = get_dataloader(
        csv_path=dataset_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,  # Shuffle for training
        use_line_map=False,  # Set to True if using line maps
        image_size=(IMAGE_SIZE, IMAGE_SIZE)  # Resize to square images
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} samples")
    print(f"Batch size: {BATCH_SIZE}, Batches per epoch: {len(train_loader)}")
    
    # Split dataset (we'll do this next)
    # Implement training loop
    # Implement validation loop
    # Implement basic loss
    # Save model




