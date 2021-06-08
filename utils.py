import torch 
import torchvision 
from torch.utils.data import DataLoader

from dataset import CityScapeDataset 


def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True 
    ):
    
    train_dataset = CityScapeDataset(
        image_dir = train_dir, 
        mask_dir = train_mask_dir,
        transform = train_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_dataset = CityScapeDataset(
        image_dir = val_dir, 
        mask_dir=val_mask_dir,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader, val_loader

