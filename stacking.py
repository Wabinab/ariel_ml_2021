"""
Stacking Machine Learning Algorithm.
Created on: 18 June 2021.
"""
import os
import numpy as np
import torch
from utils import ArielMLDataset, BaselineConv, ChallengeMetric, Baseline, simple_transform
from train_baseline import train
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import pathlib


project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
lc_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/params_train"

prefix = "50"

# training parameters
train_size = 120000
val_size = 5600
epochs = 10
save_from = 5

# hyper-parameters
H1 = 256
H2 = 1024
H3 = 256
H4 = 256

n_wavelengths = 55

main = True


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    error_dataset = None

    # First-level model training. Will train parallel so there will be another training going 
    # on at another vm, then it will upload to google cloud bucket, then main point will fetch it from
    # gcp bucket and use it. 
    for i in range(4):
        start = i * 50
        stop = (i + 1) * 50

        dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   start=start, stop=stop, error_dataset=error_dataset)

        dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 start=start, stop=stop, error_dataset=error_dataset)

        batch_size = 50

        baseline = Baseline(H1=H1, H2=H2, H3=H3, input_dim=50*n_wavelengths)

        train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, baseline)

        np.savetxt(project_dir / f'outputs/train_losses_{i}.txt',
               np.array(train_losses))
        np.savetxt(project_dir / f'outputs/val_losses_{i}.txt', np.array(val_losses))
        np.savetxt(project_dir / f'outputs/val_scores_{i}.txt', np.array(val_scores))
        torch.save(baseline, project_dir / f'outputs/model_state_{i}.pt')


    if main is True:
        # Run the second layer model training. 

        pass