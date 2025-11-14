import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from tqdm import tqdm

from unet import UNet
from dataset import get_dataloader

if __name__ == "__main__":
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


    # Load dataset
    train_loader = get_dataloader(
        csv_path=DATASET_PATH,
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




