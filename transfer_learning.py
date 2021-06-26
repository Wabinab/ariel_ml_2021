from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet

from utils import ArielMLDataset, BaselineLSTM, ChallengeMetric, Baseline, simple_transform

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


def train_model(model, criterion, opt, scheduler, loader_train, loader_val, 
                device="cpu", num_epochs=25):
    """
    :var model: (Pytorch savedmodel format, any .pt, .pth, etc) The model to use for transfer learning. 
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    val_scores = np.zeros((num_epochs, ), dtype=np.float32)
    best_val_score = 0.0
    challenge_metric = ChallengeMetric()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        val_score = 0
        model.train()

        for k, item in enumerate(tqdm(loader_train)):
            pred = model(item['lc'])
            loss = criterion(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            # train_loss += loss.detach().item()

        model.eval()

        for k, item in tqdm(enumerate(loader_val)):
            pred, _ = model(item['lc'])
            loss = criterion(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            # val_loss += loss.detach().item()
            val_score += score.detach().item()

        val_score /= len(loader_val)

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
            }, "Best_trained_model_resnet.pt")

    model.load_state_dict(best_model_wts)
    
    return val_scores, model


def main():
    # paths to data dirs
    lc_train_path = pathlib.Path(
        "/home/chowjunwei37/Documents/data/training_set/noisy_train")
    params_train_path = pathlib.Path(
        "/home/chowjunwei37/Documents/data/training_set/params_train")

    
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                    max_size=train_size, transform=simple_transform, device=device,
                                    transpose=False)
        # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                max_size=val_size, transform=simple_transform, device=device,
                                transpose=False)
    

    loader_train = DataLoader(dataset_train, batch_size=batch_size, 
                    shuffle=True, num_workers=2)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=2)

    # model = torch.hub.load("pytorch/vision:v0.9.0", "resnet34", pretrained=True)
    model = EfficientNet.from_pretrained("efficientnet-b0")


# ===================================================
# Example of transfer learning to load part of pretrained model. 
    pretrained = torchvision.models.alexnet(pretrained=True)
    class MyAlexNet(nn.Module):
        def __init__(self, my_pretrained_model):
            super(MyAlexNet, self).__init__()
            self.pretrained = my_pretrained_model
            self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                            nn.ReLU(),
                                            nn.Linear(100, 2))
        
        def forward(self, x):
            x = self.pretrained(x)
            x = self.my_new_layers(x)
            return x

    my_extended_model = MyAlexNet(my_pretrained_model=pretrained)
    my_extended_model

