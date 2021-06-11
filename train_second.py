"""
Training files compiled-ready for sending to GCP Vertex AI. 
Created on: 11 June 2021.
"""
import argparse
import hypertune
import numpy as np
import pathlib
import torch
from utils import ArielMLDataset, ChallengeMetric, Baseline, simple_transform
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm


def get_args():
    """
    Argument parser. 
    Returns: (dict) Arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--H1", help="first layer of Dense model")
    parser.add_argument("--H2", help="second layer of Dense model")
    parser.add_argument("--H3", help="third layer of Dense model")
    parser.add_argument("--D1", help="Dropout probability")
    parser.add_argument("--mean", help="mean to subtract from for transformation")
    parser.add_argument("--std", help="std dev to divide from for transformation")
    parser.add_argument("--lr", help="learning rate of Adam optimizer")
    parser.add_argument("--batch_size", help="determine batch size")

    return parser.parse_args()


def train(args, dataset_train, dataset_val):
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

    for epoch in range(1, 1+2):  # 2 epochs hardcoded
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

        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="ariel_score", 
            metric_value=val_scores,
            global_step=2  # again, hardcoded epochs number. 
        )


def main():
    args = get_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_size = 120000
    val_size = 5600

    

    # Training
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   mean=args.mean, std=args.std)
    # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 mean=args.mean, std=args.std)

    # Loaders
    batch_size = args.batch_size

    train(args, dataset_train, dataset_val)


if __name__ == "__main__":
    main()