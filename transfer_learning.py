from __future__ import print_function, division
import os
os.system("pip3 install efficientnet_pytorch torchsummary scikit-learn")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import gc
import glob

from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet

from utils import ArielMLDataset, TransferModel, ChallengeMetric, image_transform, FeatureModel

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import time
from tqdm import tqdm

from azureml.core import Run

# plt.ion()  # Interactive mode.

# project_dir = pathlib.Path(__file__).parent.absolute()

# train_size = 110000
# val_size = 15600
# batch_size = 25
# save_from = 1


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


def train_model_feature(model, criterion, loader_train, loader_val, 
            device=torch.device("cpu"), num_epochs=30, dir="outputs"):
    global prefix, save_from

    val_score = np.zeros((num_epochs, ), dtype=np.float32)
    best_val_score = 0.0
    challenge_metric = ChallengeMetric()

    baseline = FeatureModel().double().to(device)

    opt = torch.optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=2)

    try:
        temp = glob.glob(f"{dir}/model_state_dict.pt")
        ckpt = torch.load(temp[0])

        curr_epoch = ckpt["epoch"]
        baseline.load_state_dict(ckpt["baseline"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])

        print("Successfully load state dict.")
    except Exception:
        curr_epoch = 1

        print("Failed to load state dict. Training from scratch. ")

    for epoch in range(curr_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        val_score = 0

        baseline.train()

        for k, item in enumerate(tqdm(loader_train)):
            features = model.extract_features(item["lc"])
            pred = baseline(features.double())
            loss = criterion(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()

        baseline.eval()

        for k, item in enumerate(tqdm(loader_val)):
            features = model.extract_features(item["lc"])
            pred = baseline(features.double())
            loss = criterion(item["target"], pred)
            score = challenge_metric.score(item["target"], pred)
            
            val_score += score.detach().item()

        val_score /= len(loader_val)

        scheduler.step(val_score)
        
        run.log("val_score", val_score)

        val_scores[epoch - 1] = val_score

        print(f"Val score: {round(val_score, 2)}")

        if epoch >= save_from and val_score > best_val_score:
            print("Current best val score: ", val_score)
            best_val_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch": epoch,
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scheduler2": scheduler2.state_dict(),
                "model": model.state_dict()
            }, f"{dir}/model_state_dict.pt")

            torch.save(model, f'{dir}/model_state_{prefix}.pt')

        np.savetxt(f'outputs/val_scores_{prefix}.txt', np.array(val_scores))

        gc.collect()

    return val_scores, model
        


def train_model(model, criterion, loader_train, loader_val, 
                device="cpu", num_epochs=25, dir="outputs"):
    """
    :var model: (Pytorch savedmodel format, any .pt, .pth, etc) The model to use for transfer learning. 
    """
    global prefix, save_from

    best_model_wts = copy.deepcopy(model.state_dict())
    val_scores = np.zeros((num_epochs, ), dtype=np.float32)
    best_val_score = 0.0
    challenge_metric = ChallengeMetric()

    # opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", patience=2)
    scheduler2 = torch.optim.lr_scheduler.StepLR(opt, num_epochs // 2)

    try:
        temp = glob.glob(f"{dir}/Best_trained_model_efnet.pt")
        ckpt = torch.load(temp[0])

        curr_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scheduler2.load_state_dict(ckpt["scheduler2"])

        print(f"Confirmed loaded state dict at {curr_epoch}.")
    except Exception:
        curr_epoch = 1

        print("Failed to load state dict. Training from scratch. ")

    model.freeze()

    for epoch in range(curr_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        val_score = 0

        if epoch == ((num_epochs // 2) + 1):
            model.unfreeze()

        # Implement azureml.core.Run method to log val score. If not required, use otherwise. 
        model.train()

        for k, item in enumerate(tqdm(loader_train)):
            pred = model(item['lc'])
            loss = criterion(item['target'], pred)
            opt.zero_grad()
            loss.backward()

            # Try gradient clipping. 
            nn.utils.clip_grad_value_(model.parameters(), 1e-4)

            opt.step()
            # train_loss += loss.detach().item()

        model.eval()

        for k, item in enumerate(tqdm(loader_val)):
            pred = model(item['lc'])
            loss = criterion(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            # val_loss += loss.detach().item()
            val_score += score.detach().item()

        val_score /= len(loader_val)

        run.log("val_score", val_score)

        # Scheduler step
        scheduler.step(val_score)
        scheduler2.step()

        print('Val score', round(val_score, 2))

        val_scores[epoch - 1] = val_score

        if epoch >= save_from and val_score > best_val_score:
            print("Current best val score: ", val_score)
            best_val_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch": epoch,
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scheduler2": scheduler2.state_dict(),
                "model": model.state_dict()
            }, f"{dir}/Best_trained_model_efnet.pt")

            torch.save(model, f'{dir}/model_state_{prefix}.pt')

        np.savetxt(f'outputs/val_scores_{prefix}.txt', np.array(val_scores))

        gc.collect()

    model.load_state_dict(best_model_wts)
    
    return val_scores, model


def main(args):
    # paths to data dirs
    # lc_train_path = pathlib.Path("/home/chowjunwei37/Documents/data/training_set/noisy_train")
    # params_train_path = pathlib.Path("/home/chowjunwei37/Documents/data/training_set/params_train")

    global prefix, train_size, val_size, batch_size, save_from

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(device)

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        print("Cannot set multiprocessing spawn. ")

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # lc_train_path = pathlib.Path("./training_set/noisy_train")
    # params_train_path = pathlib.Path("./training_set/params_train")


    # lc_train_path = "./training_set/noisy_train"
    # lc_train_path = "/home/chowjunwei37/Documents/data/training_set/aug_noisy_train_img"
    lc_train_path = args.ds_ref
    params_train_path = pd.read_csv("./data/training_set_params_aug.csv")
    
    dataset_train = ArielMLDataset(lc_train_path, "train", params_train_path, shuffle=True, start_ind=0,
                                    max_size=train_size, transform=image_transform, device=device,
    )
        # Validation
    dataset_val = ArielMLDataset(lc_train_path, "val", params_train_path, shuffle=True, start_ind=train_size,
                                max_size=val_size, transform=image_transform, device=device,
                                )
    

    loader_train = DataLoader(dataset_train, batch_size=batch_size, 
                    shuffle=True, pin_memory=False)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, pin_memory=False)

    # loader_train = DeviceDataLoader(loader_train, device)
    # loader_val = DeviceDataLoader(loader_val, device)

    # model = torch.hub.load("pytorch/vision:v0.9.0", "resnet34", pretrained=True)
    model_name = "efficientnet-b1"
    image_size = 224

    model = EfficientNet.from_pretrained(model_name)
    # model = TransferModel(model_name, model, image_size)
    model = to_device(model, device)

    criterion = MSELoss()

    val_scores, model = train_model_feature(model, criterion, loader_train, loader_val, device, num_epochs=30)

    np.savetxt(f'outputs/val_scores_{prefix}.txt', np.array(val_scores))
    torch.save(model, f'outputs/model_state_{prefix}.pt')

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    prefix = "transfer_learning"
    train_size = 6000
    val_size = 2500
    batch_size = 25
    save_from = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, dest="ds_ref")
    args = parser.parse_args()

    run = Run.get_context()

    main(args)