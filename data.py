import os
import time
import random
from collections import OrderedDict

import numpy as np
from PIL import Image
import scipy.io
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def get_metadata(name):
    if name == "mnist":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
            }
        )
    elif name == "mnist_m":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "cifar10":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": 50000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "melanoma":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 2,
                "train_images": 33126,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    elif name == "afhq":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 3,
                "train_images": 14630,
                "val_images": 1500,
                "num_channels": 3,
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 4,
                "train_images": 109036,
                "val_images": 12376,
                "num_channels": 3,
            }
        )
    elif name == "cars":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 196,
                "train_images": 8144,
                "val_images": 8041,
                "num_channels": 3,
            }
        )
    elif name == "flowers":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 102,
                "train_images": 2040,
                "val_images": 6149,
                "num_channels": 3,
            }
        )
    elif name == "gtsrb":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 43,
                "train_images": 39252,
                "val_images": 12631,
                "num_channels": 3,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata


class oxford_flowers_dataset(Dataset):
    def __init__(self, indexes, labels, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for i in indexes:
            self.images.append(
                os.path.join(
                    root_dir,
                    "jpg",
                    "image_" + "".join(["0"] * (5 - len(str(i)))) + str(i) + ".jpg",
                )
            )
            self.targets.append(labels[i - 1] - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name == "mnist_m":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name in ["imagenette", "melanoma", "afhq"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize(74),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cars":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "flowers":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        splits = scipy.io.loadmat(os.path.join(data_dir, "setid.mat"))
        labels = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
        labels = labels["labels"][0]
        train_set = oxford_flowers_dataset(
            np.concatenate((splits["trnid"][0], splits["valid"][0]), axis=0),
            labels,
            data_dir,
            transform_train,
        )
    elif name == "gtsrb":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set

class CustomMNISTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        # Transpose to [80, 1, 28, 28]
        self.images = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32, device='cpu')
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long, device='cpu')
        else:
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.labels is None:
            label = None
        else:
            label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, 1
    
class SynthesizedDataset(Dataset):
    def __init__(self, images, labels=None):
        super(SynthesizedDataset, self).__init__()
        self.images = torch.from_numpy(images).view(-1, 1, 28, 28).to('cpu').float() / 255.0
        assert self.images.min().item() >= 0 and self.images.max().item() <= 1
        if labels is not None:
            self.labels = torch.tensor(labels).view(-1).to('cpu')
        else:
            labels = np.random.randint(10, size=len(self.images))
            self.labels = torch.tensor(labels).view(-1).to('cpu')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def skewed_mnist(args, minority_classes=[0,1,2,3,4], minority_count=100, majority_count=1000, val_per_class=20):
    # Define your transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load full MNIST dataset
    full_train_set = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Load full MNIST test set for validation
    full_test_set = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_targets = full_train_set.targets
    test_targets = full_test_set.targets

    train_indices = []
    val_indices = []

    for cls in range(10):
        # Get all indices for the current class
        cls_train_indices = (train_targets == cls).nonzero(as_tuple=True)[0].tolist()
        cls_test_indices = (test_targets == cls).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(cls_train_indices)
        random.shuffle(cls_test_indices)

        # Select training samples
        if cls in minority_classes:
            train_count = minority_count
        else:
            train_count = majority_count
        train_indices.extend(cls_train_indices[:train_count])

        # Select validation samples
        val_indices.extend(cls_test_indices[:val_per_class])

    # Create skewed train and validation datasets
    skewed_train_set = Subset(full_train_set, train_indices)
    val_set = Subset(full_test_set, val_indices)

    return skewed_train_set, val_set


def leave_out_classes_random(dataset, leave_out_classes, max_per_class=None, seed=None):
    """
    Removes a random subset of samples from specified classes.

    Args:
        dataset (Subset or Dataset): PyTorch dataset or subset.
        leave_out_classes (list or set): Class labels to remove.
        max_per_class (int or None): Max number of samples to remove per class. If None, removes all.
        seed (int or None): Random seed for reproducibility.

    Returns:
        Subset: New dataset with specified minority samples removed.
    """
    # Use current time as seed if not provided
    if seed is None:
        seed = int(time.time())
    random.seed(seed)

    # Get access to full dataset and target labels
    if isinstance(dataset, Subset):
        data_source = dataset.dataset
        subset_indices = dataset.indices
        targets = [data_source[i][1] for i in subset_indices]
    else:
        data_source = dataset
        subset_indices = list(range(len(dataset)))
        targets = [data_source[i][1] for i in subset_indices]

    # Map: class → list of (i, idx)
    class_to_indices = {cls: [] for cls in leave_out_classes}
    for i, (idx, label) in enumerate(zip(subset_indices, targets)):
        if label in leave_out_classes:
            class_to_indices[label].append((i, idx))

    # Randomly select which to remove
    to_remove = set()
    for cls, samples in class_to_indices.items():
        random.shuffle(samples)
        num_to_remove = len(samples) if max_per_class is None else min(max_per_class, len(samples))
        to_remove.update(i for i, _ in samples[:num_to_remove])

    # Keep the rest
    remaining_indices = [idx for i, idx in enumerate(subset_indices) if i not in to_remove]

    return Subset(data_source, remaining_indices)



def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
