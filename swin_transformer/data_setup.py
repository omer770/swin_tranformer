import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Sampler
from typing  import  Tuple, Dict, List

NUM_WORKERS = os.cpu_count()


import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class WeightedSampler(data.sampler.Sampler):
    """Weighted random sampler for imbalanced datasets in PyTorch.

    Args:
        data_source (Dataset): Dataset containing the samples.
        weights (Tensor): Tensor containing the weights for each sample.
        num_samples (int, optional): Number of samples to draw. Defaults to `len(data_source)`.
        replacement (bool, optional): Whether to sample with replacement. Defaults to `False`.

    Raises:
        ValueError: If `weights` is not a 1D tensor or if `num_samples` exceeds `len(data_source)` when `replacement` is `False`.
    """

    def __init__(self, data_source, weights, num_samples=None, replacement=False):
        if not isinstance(weights, torch.Tensor):
            raise ValueError("Weights must be a torch.Tensor.")
        if weights.dim() != 1:
            raise ValueError("Weights must be a 1D tensor.")
        if num_samples is None:
            num_samples = len(data_source)
        if not replacement and num_samples > len(data_source):
            raise ValueError("If replacement is False, num_samples cannot be greater than len(data_source).")
        self.data_source = data_source
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        samples = torch.multinomial(self.weights, self.num_samples, self.replacement).tolist()
        print("Sampled indices:", samples)
        return iter(samples)

    def __len__(self):
        return self.num_samples

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                 test_dir=path/to/test_dir,
                                 transform=some_transform,
                                 batch_size=32,
                                 num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes
    labels = [label for _, label in train_data]
    class_counts = torch.bincount(torch.tensor(labels))
    sample_weights = [1/class_counts[i] for i in labels]

    weighted_sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(train_data), replacement=True)


    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=weighted_sampler,  # Use weighted sampler
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names