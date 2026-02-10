"""
Utility for loading the sequential CIFAR-10 dataset as a torch Dataloader with JAX arrays.
"""

from typing import Tuple
import jax
import torch
from jax import random as jr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, train: bool = True, permutation=None):
        self.cifar10_data = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.permutation = permutation

    def __len__(self):
        return len(self.cifar10_data)

    def __getitem__(self, idx):
        image, label = self.cifar10_data[idx]
        # Flatten the image to a sequence of 1024 pixels (32x32)
        image = image.numpy().reshape(-1)  # Shape (1024,)
        if self.permutation is not None:
            image = image[self.permutation]  # Apply permutation if provided
        return image, label
    

def get_cifar10_dataloader(batch_size: int, train: bool = True, val_split=0.1, permutation=None) -> DataLoader | Tuple[DataLoader, DataLoader]:
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack(images)[..., None]  # Shape (batch_size, 1024, 1)
        labels = np.array(labels)  # Shape (batch_size,)
        return jax.device_put(images), jax.device_put(labels)
    dataset = CIFAR10Dataset(train=train, permutation=permutation)
    # Optionally split the dataset into training and validation sets
    if val_split > 0 and train:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        return train_loader, val_loader
    
    # Else create a DataLoader that returns JAX arrays
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, collate_fn=collate_fn)
    return dataloader

