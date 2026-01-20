import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from libcll.datasets.cl_base_dataset import CLBaseDataset
from libcll.datasets.utils import get_transition_matrix
from libcll.datasets.augmentation import get_augmentation, get_test_transform


class CLTiny200(CLBaseDataset):
    """
    TinyImageNet dataset with support for synthetic complementary labels.

    TinyImageNet has 200 classes, each with 500 training images, 50 validation images,
    and 50 test images. Images are 64x64 pixels.

    Parameters
    ----------
    root : str
        path to store dataset file.

    train : bool
        training set if True, else testing set (validation set).

    transform : callable, optional
        a function/transform that takes in a PIL image and returns a transformed version.

    target_transform : callable, optional
        a function/transform that takes in the target and transforms it.

    download : bool
        if true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.

    num_cl : int
        the number of complementary labels for each data point.

    Attributes
    ----------
    data : Tensor
        the feature of sample set.

    targets : Tensor
        the complementary labels for corresponding sample (or true labels before gen_complementary_target is called).

    true_targets : Tensor
        the ground-truth labels for corresponding sample.

    num_classes : int
        the number of classes (200 for TinyImageNet).

    input_dim : int
        the feature space after data compressed into a 1D dimension.

    """

    def __init__(
        self,
        root="../data/tiny200/tiny-imagenet-200",
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        num_cl=1,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 200
        self.input_dim = 3 * 64 * 64

        if download:
            self._download()

        # Load class names
        self.class_to_idx = self._load_class_names()
        self.classes = list(self.class_to_idx.keys())

        # Load data
        self.data, self.targets = self._load_data()
        self.targets = torch.Tensor(self.targets)
        self.true_targets = self.targets.clone()

    def _download(self):
        """Download TinyImageNet dataset if not already present."""
        if os.path.exists(self.root) and os.path.isdir(self.root):
            # Check if the dataset is already downloaded
            if os.path.exists(os.path.join(self.root, 'wnids.txt')):
                print(f"Dataset already exists at {self.root}")
                return
        
        print(f"TinyImageNet dataset not found at {self.root}.")
        print("Please download it manually from http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print(f"and extract it to {self.root}")
        raise FileNotFoundError(
            f"TinyImageNet dataset not found at {self.root}. "
            "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        )

    def _load_class_names(self):
        """Load class names and create mapping."""
        wnids_path = os.path.join(self.root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(
                f"wnids.txt not found at {wnids_path}. "
                "Make sure the TinyImageNet dataset is properly extracted."
            )
        
        with open(wnids_path, 'r') as f:
            class_ids = [line.strip() for line in f.readlines()]
        
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
        return class_to_idx

    def _load_data(self):
        """Load images and labels."""
        images = []
        labels = []

        if self.train:
            # Load training data
            train_dir = os.path.join(self.root, 'train')
            for class_id in self.classes:
                class_dir = os.path.join(train_dir, class_id, 'images')
                if not os.path.exists(class_dir):
                    continue
                
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        images.append(img_path)
                        labels.append(self.class_to_idx[class_id])
        else:
            # Load validation data
            val_dir = os.path.join(self.root, 'val')
            val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
            
            if not os.path.exists(val_annotations_path):
                raise FileNotFoundError(
                    f"Validation annotations not found at {val_annotations_path}"
                )
            
            with open(val_annotations_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_id = parts[1]
                    img_path = os.path.join(val_dir, 'images', img_name)
                    images.append(img_path)
                    labels.append(self.class_to_idx[class_id])

        return images, labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the complementary label(s).
        """
        img_path = self.data[index]
        target = self.targets[index]

        # Load image
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @classmethod
    def build_dataset(cls, dataset_name=None, train=True, num_cl=0, 
                     transition_matrix=None, noise=None, seed=1126, data_augment="flipflop"):
        """
        Build TinyImageNet dataset with appropriate transforms and complementary labels.

        Parameters
        ----------
        dataset_name : str, optional
            name of the dataset (not used, kept for compatibility)
        
        train : bool
            if True, return training set, else return validation set
        
        num_cl : int
            number of complementary labels to generate
        
        transition_matrix : str, optional
            type of transition matrix for generating complementary labels
        
        noise : float, optional
            noise level for transition matrix
        
        seed : int
            random seed for reproducibility
        
        data_augment : str, optional
            type of data augmentation to use (default: "flipflop")
            Options: "flipflop", "autoaug", "randaug", "cutout"

        Returns
        -------
        dataset : TinyImageNet
            TinyImageNet dataset instance
        """
        if train:
            train_transform = get_augmentation(
                augment_type=data_augment,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                image_size=64
            )
            dataset = cls(
                train=True,
                transform=train_transform,
                num_cl=num_cl,
            )
            # Generate synthetic complementary labels
            Q = get_transition_matrix(transition_matrix, "tiny200", 
                                     dataset.num_classes, noise, seed)
            dataset.gen_complementary_target(num_cl, Q)
        else:
            test_transform = get_test_transform(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            dataset = cls(
                train=False,
                transform=test_transform,
                num_cl=num_cl,
            )
        return dataset
