from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import gc

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet

from utils import ArielMLDataset, TransferModel, ChallengeMetric, simple_transform

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
batch_size = 25
save_from = 1


def to_device(data, device):
    """
    Taken from https://www.kaggle.com/veb101/transfer-learning-using-efficientnet-models
    for moving tensor(s) to chosen device. 

    :var data: (Tensors) Can be tensors or can be model (weights). 
    :var device: Device to move to. In this case, GPU. 
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]  # Recursive call. 
    return data.to(device, non_blocking=True)


# Also taken from https://www.kaggle.com/veb101/transfer-learning-using-efficientnet-models
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def train_model(model, criterion, loader_train, loader_val, 
                device="cpu", num_epochs=25, dir="./outputs"):
    """
    :var model: (Pytorch savedmodel format, any .pt, .pth, etc) The model to use for transfer learning. 
    """
    if device != "cpu":
        torch.cuda.empty_cache()

    best_model_wts = copy.deepcopy(model.state_dict())
    val_scores = np.zeros((num_epochs, ), dtype=np.float32)
    best_val_score = 0.0
    challenge_metric = ChallengeMetric()

    opt = torch.optim.SGD(model.parameters())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=2)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        val_score = 0

        # Implement azureml.core.Run method to log val score. If not required, use otherwise. 
        model.train()

        for k, item in enumerate(tqdm(loader_train)):
            pred = model(item['lc'])
            loss = criterion(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # train_loss += loss.detach().item()

        model.eval()

        for k, item in enumerate(tqdm(loader_val)):
            pred, _ = model(item['lc'])
            loss = criterion(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            # val_loss += loss.detach().item()
            val_score += score.detach().item()

        val_score /= len(loader_val)

        scheduler.step(val_score)

        print('Val score', round(val_score, 2))

        val_scores[epoch] = val_score

        if epoch >= save_from and val_score > best_val_score:
            best_val_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch": epoch,
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": model.state_dict()
            }, f"{dir}/Best_trained_model_resnet.pt")

        gc.collect()

    model.load_state_dict(best_model_wts)
    
    return val_scores, model


def main():
    # paths to data dirs
    # lc_train_path = pathlib.Path("/home/chowjunwei37/Documents/data/training_set/noisy_train")
    # params_train_path = pathlib.Path("/home/chowjunwei37/Documents/data/training_set/params_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lc_train_path = pathlib.Path("./training_set/noisy_train")
    params_train_path = pathlib.Path("./training_set/params_train")
    
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                    max_size=train_size, transform=simple_transform, device=device,
                                    transpose=False)
        # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                max_size=val_size, transform=simple_transform, device=device,
                                transpose=False)
    

    loader_train = DataLoader(dataset_train, batch_size=batch_size, 
                    shuffle=True, num_workers=3)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=2)

    # model = torch.hub.load("pytorch/vision:v0.9.0", "resnet34", pretrained=True)
    model_name = "efficientnet-b0"
    image_size = 224

    model = EfficientNet.from_pretrained(model_name)
    model = TransferModel(model_name, model, image_size)
    model = to_device(model, device)

    criterion = MSELoss()

    val_scores, model = train_model(model, criterion, loader_train, loader_val, device, 10)




