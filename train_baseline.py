"""Define and train the baseline model"""
import numpy as np
import torch
from torch._C import device
from utils import ArielMLDataset, BaselineLSTMAlt, ChallengeMetric, Baseline, simple_transform
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import pathlib
import os
import glob


__author__ = "Mario Morvan"
__email__ = "mario.morvan.18@ucl.ac.uk"

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
lc_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/params_train"

# lc_train_path = project_dir / \
#      "/home/dsvm113/IdeaProjects/workspace/data_1/training_set/noisy_train"
# params_train_path = project_dir / \
#      "/home/dsvm113/IdeaProjects/workspace/data_1/training_set/params_train"

prefix = "alt_lstm"

# training parameters
train_size = 120000
val_size = 5600
epochs = 70
save_from = 1

# hyper-parameters
# H1 = 256
# H2 = 1024
# H3 = 256
# H_LSTM = 512

H1 = 256
H2 = 512
H_LSTM = 128


# -------------------------------------------------

def train(batch_size, dataset_train, dataset_val, device):

    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Define baseline model
    # baseline = Baseline(H1=H1, H2=H2, H3=H3).double().to(device)
    baseline = BaselineLSTMAlt(H_LSTM, H1, H2, batch_size=batch_size, device=device).double().to(device)

    # Define Loss, metric and optimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()
    opt = Adam(baseline.parameters())

    try:
        temp = glob.glob("outputs/model_continue_train*.pt")
        ckpt = torch.load(temp[0])

        curr_epoch = ckpt["epoch"]
        baseline.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])

        print(f"Confirm loaded state dict at epoch {curr_epoch}.")
    except Exception:
        curr_epoch = 1

        print("Failed to load state dict or it did not exist. Training from scratch.")

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

        np.savetxt(project_dir / f'outputs/train_losses_{prefix}.txt',
            np.array(train_losses))
        np.savetxt(project_dir / f'outputs/val_losses_{prefix}.txt', np.array(val_losses))
        np.savetxt(project_dir / f'outputs/val_scores_{prefix}.txt', np.array(val_scores))

        if epoch >= save_from and val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "state_dict": baseline.state_dict(),
                "opt": opt.state_dict(),
            }, f"outputs/model_continue_train_{prefix}.pt")
            torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')

    return train_losses, val_losses, val_scores, baseline


# -------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # torch.set_num_interop_threads(os.cpu_count() - 2)
    torch.set_num_threads(os.cpu_count() - 1)
    # torch.set_num_threads(15)


    # Training
    dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                max_size=train_size, transform=simple_transform, device=device,
                                transpose=True)
    # Validation
    dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                max_size=val_size, transform=simple_transform, device=device,
                                transpose=True)

    # Loaders
    # batch_size = int(train_size / 4)
    batch_size = 50
    
    train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, device)
    
    np.savetxt(project_dir / f'outputs/train_losses_{prefix}.txt',
            np.array(train_losses))
    np.savetxt(project_dir / f'outputs/val_losses_{prefix}.txt', np.array(val_losses))
    np.savetxt(project_dir / f'outputs/val_scores_{prefix}.txt', np.array(val_scores))
    torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')


if __name__ == '__main__':
    main()

