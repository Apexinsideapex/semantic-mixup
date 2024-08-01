import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CutMix(object):
    def __init__(self, alpha, prob):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        images, labels = batch
        if np.random.rand() > self.prob:
            return images, labels
        
        batch_size = len(images)
        rand_index = torch.randperm(batch_size)

        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return images, (labels, labels[rand_index], lam)
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class DatasetFactory:
    @staticmethod
    def get_dataset(name, train=True):
        if name == "cub200":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/CUB_200_2011_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        elif name == "stanford_dogs":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/Stanford_Dogs_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        elif name == "cifar10":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/CIFAR10_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def get_transform(train, dataset_name):
        if dataset_name == "cifar10":
            if train:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:  # For CUB200 and Stanford Dogs
            if train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

def get_dataloader(dataset, batch_size, num_workers, shuffle=True, use_cutmix=False, cutmix_alpha=1.0, cutmix_prob=0.5):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    
    if use_cutmix:
        return CutMixLoader(loader, cutmix_alpha, cutmix_prob)
    return loader

class CutMixLoader:
    def __init__(self, loader, alpha, prob):
        self.loader = loader
        self.cutmix = CutMix(alpha, prob)

    def __iter__(self):
        for batch in self.loader:
            yield self.cutmix(batch)

    def __len__(self):
        return len(self.loader)