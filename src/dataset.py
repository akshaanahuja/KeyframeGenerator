"""
PyTorch Dataset for Frame Interpolation

Loads I0, It (ground truth), I1 triplets from CSV for training.
Optionally includes line maps for conditional generation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


class FrameInterpolationDataset(Dataset):
    """
    Dataset for frame interpolation training.
    
    Loads triplets of (I0, It, I1) frames from CSV.
    Optionally includes line maps for conditional generation.
    """
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        use_line_map: bool = False,
        image_size: tuple = None
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with columns: path_i0, path_it, path_i1, path_ltau
            transform: Optional torchvision transform to apply
            use_line_map: Whether to load and return line maps
            image_size: Optional (H, W) to resize images. If None, uses original size.
        """
        self.df = pd.read_csv(csv_path)
        self.use_line_map = use_line_map
        self.image_size = image_size
        
        # Default transform: convert to tensor and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform
        
        # Transform for line maps (grayscale, no normalization)
        self.line_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        # Add resize if specified
        if image_size:
            resize = transforms.Resize(image_size)
            self.transform = transforms.Compose([resize, self.transform])
            self.line_transform = transforms.Compose([resize, self.line_transform])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            dict with keys:
                - 'i0': Previous frame tensor (3, H, W)
                - 'i1': Next frame tensor (3, H, W)
                - 'it': Ground truth middle frame tensor (3, H, W)
                - 'line_map': Optional line map tensor (1, H, W) if use_line_map=True
        """
        row = self.df.iloc[idx]
        
        # Load RGB images
        i0 = Image.open(row['path_i0']).convert('RGB')
        it = Image.open(row['path_it']).convert('RGB')  # Ground truth
        i1 = Image.open(row['path_i1']).convert('RGB')
        
        # Apply transforms
        i0 = self.transform(i0)
        it = self.transform(it)
        i1 = self.transform(i1)
        
        result = {
            'i0': i0,
            'i1': i1,
            'it': it,  # Ground truth target
        }
        
        # Optionally load line map
        if self.use_line_map and pd.notna(row.get('path_ltau', None)):
            line_map_path = row['path_ltau']
            if Path(line_map_path).exists():
                line_map = Image.open(line_map_path).convert('L')  # Grayscale
                line_map = self.line_transform(line_map)  # (1, H, W)
                result['line_map'] = line_map
            else:
                # Create dummy line map if file doesn't exist
                result['line_map'] = torch.zeros(1, *i0.shape[1:])
        elif self.use_line_map:
            # No line map path provided, create zeros
            result['line_map'] = torch.zeros(1, *i0.shape[1:])
        
        return result


def get_dataloader(
    csv_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    use_line_map: bool = False,
    image_size: tuple = None,
    **kwargs
):
    """
    Convenience function to create a DataLoader.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        use_line_map: Whether to load line maps
        image_size: Optional (H, W) to resize images
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = FrameInterpolationDataset(
        csv_path=csv_path,
        use_line_map=use_line_map,
        image_size=image_size
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )

