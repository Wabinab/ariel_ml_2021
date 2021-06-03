"""Define and train the baseline model"""
import numpy as np
import torch
from utils import ArielMLDataset, ChallengeMetric, Baseline, simple_transform
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import pathlib


__author__ = "Mario Morvan"
__email__ = "mario.morvan.18@ucl.ac.uk"

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
lc_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/params_train"

prefix = "4"

# training parameters
train_size = 120000
val_size = 5600
epochs = 40
save_from = 20

# hyper-parameters
H1 = 256
H2 = 1024
H3 = 256


# -------------------------------------------------

def train(batch_size, dataset_train, dataset_val):
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Define baseline model
    baseline = Baseline(H1=H1, H2=H2, H3=H3).double().to(device)

    # Define Loss, metric and optimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()
    opt = Adam(baseline.parameters())

    # Lists to record train and val scores
    train_losses = []
    val_losses = []
    val_scores = []
    best_val_score = 0.

    for epoch in range(1, 1+epochs):
        print("epoch", epoch)
        train_loss = 0
        val_loss = 0
        val_score = 0
        baseline.train()

        for k, item in tqdm(enumerate(loader_train)):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
        train_loss = train_loss / len(loader_train)
        baseline.eval()

        for k, item in tqdm(enumerate(loader_val)):
            pred = baseline(item['lc'])
            loss = loss_function(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            val_loss += loss.detach().item()
            val_score += score.detach().item()

        val_loss /= len(loader_val)
        val_score /= len(loader_val)

        print('Training loss', round(train_loss, 6))
        print('Val loss', round(val_loss, 6))
        print('Val score', round(val_score, 2))

        train_losses += [train_loss]
        val_losses += [val_loss]
        val_scores += [val_score]

        if epoch >= save_from and val_score > best_val_score:
            best_val_score = val_score
            torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')

    return train_losses, val_losses, val_scores, baseline


# -------------------------------------------------

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Training
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device)
    # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device)

    # Loaders
    # batch_size = int(train_size / 4)
    batch_size = 100
    
    train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val)
    
    np.savetxt(project_dir / f'outputs/train_losses_{prefix}.txt',
               np.array(train_losses))
    np.savetxt(project_dir / f'outputs/val_losses_{prefix}.txt', np.array(val_losses))
    np.savetxt(project_dir / f'outputs/val_scores_{prefix}.txt', np.array(val_scores))
    torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')
