"""
Training files compiled-ready for sending to GCP Vertex AI. 
Created on: 11 June 2021.
"""
import argparse
# import hypertune
import numpy as np
import pathlib
import sys
import torch
from utils import ArielMLDataset, ChallengeMetric, Baseline, simple_transform
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm

import wandb

hparam_defaults = dict(
    H1 = 128,
    H2 = 256,
    H3 = 128,
    D1 = 0.1, 
    mean = 1.0,
    std = 0.04,
    lr = 1e-4,
    batch_size = 100
)


def get_args(args):
    """
    Argument parser. 
    Returns: (dict) Arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--H1", type=int, help="first layer of Dense model")
    parser.add_argument("--H2", type=int, help="second layer of Dense model")
    parser.add_argument("--H3", type=int, help="third layer of Dense model")
    parser.add_argument("--D1", type=float, help="Dropout probability")
    parser.add_argument("--mean", type=float, help="mean to subtract from for transformation")
    parser.add_argument("--std", type=float, help="std dev to divide from for transformation")
    parser.add_argument("--lr", type=float, help="learning rate of Adam optimizer")
    parser.add_argument("--batch_size", type=int, help="determine batch size")

    return parser.parse_args(args)


def train(args, dataset_train, dataset_val, device):
    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    # Define baseline model
    baseline = Baseline(H1= args.H1, H2= args.H2, H3= args.H3, D1=args.D1).double().to(device)

    # Define Loss, metric and optimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()
    opt = Adam(baseline.parameters(), lr=args.lr)

    # Lists to record train and val scores
    val_scores = []

    for epoch in range(1, 1+25):  # 2 epochs hardcoded
        val_score = 0
        baseline.train()

        for k, item in tqdm(enumerate(loader_train)):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
        baseline.eval()

        for k, item in tqdm(enumerate(loader_val)):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            val_score += score.detach().item()

        val_score /= len(loader_val)
        val_scores += [val_score]

        wandb.log({"ariel_score": val_scores})


def main():
    args1 = get_args(sys.argv[1:])

    wandb.init(config=args1)
    args = wandb.config

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_size = 2000
    val_size = 1000

    # Training
    dataset_train = ArielMLDataset(None, params_path="./", shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   mean=args.mean, std=args.std)
    # Validation
    dataset_val = ArielMLDataset(None, params_path="./", shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 mean=args.mean, std=args.std)

    # Loaders
    train(args, dataset_train, dataset_val, device)


if __name__ == "__main__":
    main()