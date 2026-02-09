"""
Utility for loading the sequential MNIST dataset as a torch Dataloader with JAX arrays.
"""

import jax
from jax import random as jr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SMNISTDataset(Dataset):
    def __init__(self, train: bool = True, permutation=None):
        self.mnist_data = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.permutation = permutation

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        # Flatten the image to a sequence of 784 pixels
        image = image.numpy().reshape(-1)  # Shape (784,)
        if self.permutation is not None:
            image = image[self.permutation]  # Apply permutation if provided
        return image, label
    
def get_smnist_dataloader(batch_size: int, train: bool = True, permutation=None) -> DataLoader:
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack(images)[..., None]  # Shape (batch_size, 784, 1)
        labels = np.array(labels)  # Shape (batch_size,)
        return jax.device_put(images), jax.device_put(labels)
    dataset = SMNISTDataset(train=train, permutation=permutation)
    # Create a DataLoader that returns JAX arrays
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, collate_fn=collate_fn)

    return dataloader

if __name__ == "__main__":
    # Example usage
    permutation = np.arange(784)  # Example permutation (identity)
    train_loader = get_smnist_dataloader(batch_size=64, train=True, permutation=permutation)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx == 1:
            break