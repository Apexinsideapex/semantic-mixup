import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import numpy as np

from config import Config
from data.datasets import DatasetFactory, get_dataloader_cutmix, get_dataloader_semcutmix, CutMix, SemCutMix
from models.models import ModelFactory
from utils.utils import set_seed, accuracy
import time


set_seed(Config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dataset_name in Config.DATASETS:
    for model_name in Config.MODELS:
        train_dataset = DatasetFactory.get_dataset(dataset_name, train=True)
        num_classes = train_dataset.num_classes
        
        train_loader = get_dataloader_cutmix(train_dataset, Config.BATCH_SIZE, Config.NUM_WORKERS, shuffle=True)

        cutmix = CutMix(alpha=Config.CUTMIX_ALPHA, prob=Config.CUTMIX_PROB)

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            if isinstance(targets, tuple):
                targets_a, targets_b, lam = targets
                targets_a, targets_b = targets_a.to(device), targets_b.to(device)
            else:
                targets = targets.to(device)
            cutmix_img, targets = cutmix((inputs, targets))
            cutmix_img = cutmix_img.squeeze(0)
            break

        cutmix_pil = tensor_to_pil(cutmix_img)
