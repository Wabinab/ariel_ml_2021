from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import time
from tqdm import tqdm

# plt.ion()  # Interactive mode. 

project_dir = pathlib.Path(__file__).parent.absolute()

train_size = 110000
val_size = 15600

dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                max_size=train_size, transform=simple_transform, device=device,
                                transpose=False)
    # Validation
dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                            max_size=val_size, transform=simple_transform, device=device,
                            transpose=False)

# paths to data dirs
lc_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/params_train"

loader_train = DataLoader(dataset_train, batch_size=batch_size, 
                shuffle=True, num_workers=3)
loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=3)

