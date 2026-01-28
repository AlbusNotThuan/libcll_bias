"""
Data augmentation strategies for various datasets.
"""
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_augmentation(augment_type="flipflop", mean=None, std=None, image_size=32):
    """
    Get augmentation transforms for various datasets.
    
    Args:
        augment_type (str): Type of augmentation. Options: "flipflop", "autoaug", "randaug", "cutout"
        mean (list): Mean for normalization
        std (list): Standard deviation for normalization
        image_size (int): Size of the image (32 for CIFAR, 64 for Tiny200)
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]
    if std is None:
        std = [0.247, 0.2435, 0.2616]
    
    padding = 4 if image_size == 32 else 8
    
    if augment_type == "flipflop":
        # Default augmentation: RandomHorizontalFlip + RandomCrop
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=padding),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    
    elif augment_type == "autoaug":
        # AutoAugment - use CIFAR10 policy for 32x32 images, IMAGENET policy for larger images
        policy = transforms.AutoAugmentPolicy.CIFAR10 if image_size == 32 else transforms.AutoAugmentPolicy.IMAGENET
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=padding),
            transforms.AutoAugment(policy),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    
    elif augment_type == "randaug":
        # RandAugment with default parameters
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=padding),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    
    elif augment_type == "cutout":
        # Cutout augmentation
        cutout_length = 16 if image_size == 32 else 32
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=padding),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(n_holes=1, length=cutout_length),
        ]
    
    else:
        raise ValueError(f"Unknown augmentation type: {augment_type}. "
                        "Choose from 'flipflop', 'autoaug', 'randaug', 'cutout'")
    
    print(f"Using augmentation: {augment_type}")
    
    return transforms.Compose(transform_list)


def get_test_transform(mean=None, std=None):
    """
    Get test/validation transforms (no augmentation).
    
    Args:
        mean (list): Mean for normalization
        std (list): Standard deviation for normalization
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]
    if std is None:
        std = [0.247, 0.2435, 0.2616]
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
