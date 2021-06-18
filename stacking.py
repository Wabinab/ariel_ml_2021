"""
Stacking Machine Learning Algorithm.
Created on: 18 June 2021.
"""
import glob
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
from sklearn.metrics import mean_squared_error


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

n_timesteps = 300
n_wavelengths = 55

main = True


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    error_dataset = None

    if main is True:
        looping = np.arange(4)
    else:
        looping = np.arange(4, 6)

    # First-level model training. Will train parallel so there will be another training going 
    # on at another vm, then it will upload to google cloud bucket, then main point will fetch it from
    # gcp bucket and use it. 
    for i in looping:
        start = i * 50
        stop = (i + 1) * 50

        dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   start=start, stop=stop)

        dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 start=start, stop=stop)

        batch_size = 50

        baseline = Baseline(H1=H1, H2=H2, H3=H3, input_dim=(start - stop) * n_wavelengths)

        train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, baseline,
                                                                epochs, save_from)

        np.savetxt(project_dir / f'outputs/train_losses_{i}.txt',
               np.array(train_losses))
        np.savetxt(project_dir / f'outputs/val_losses_{i}.txt', np.array(val_losses))
        np.savetxt(project_dir / f'outputs/val_scores_{i}.txt', np.array(val_scores))
        torch.save(baseline, project_dir / f'model/model_state_{i}.pt')


    if main is True:
        # First, calculate the errors on the TRAINING dataset. 
        # Ideally, they will be saved into one .csv file since we are flattening it anyways. 

        dataset_train_eval = ArielMLDataset(lc_train_path, shuffle=False, transform=simple_transform)

        # If the below don't run, please SET BATCH SIZE TO 1. 
        loader_train_eval = DataLoader(dataset_train_eval, batch_size=500, shuffle=False)

        train_eval_df = pd.DataFrame()
        baselines = np.array([torch.load(path_) for path_ in glob.glob("./model/*.pt")])

        for k, item in tqdm.tqdm(enumerate(loader_train_eval)):  # I actually don't know why enum here
            y_true = np.array(item["target"])

            pred = []

            for baseline in baselines:
                y_pred = np.array(baseline(item["lc"]))
            
                # difference error
                abs_error = y_pred - y_true  # we are not finding the absolute error here. Just "difference". 

                pred += abs_error

            train_eval_df[item["filename"]] = np.array(pred).flatten()

        train_eval_df.to_csv("./outputs/train_errors.csv", header=False, sep=",", index=False)


        # Start preparation for training second layer model. 

        dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   error_dataset=train_eval_df)

        dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 error_dataset=train_eval_df)

        batch_size = 50

        # Provided that "pred" has not been deleted yet, this will success
        # If it failed, it should stop immediately. No assertion had been put in place
        # which is a bad practice. 

        total_length = (n_timesteps * n_wavelengths) + np.array(pred).flatten().size
        baseline = Baseline(H1=256, H2=1024, H3=1024, H4=256, input_dim=total_length, model_num=2)

        train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, baseline,
                                                                epochs, save_from)

        prefix = "finale"

        np.savetxt(project_dir / f'outputs/train_losses_{prefix}.txt',
               np.array(train_losses))
        np.savetxt(project_dir / f'outputs/val_losses_{prefix}.txt', np.array(val_losses))
        np.savetxt(project_dir / f'outputs/val_scores_{prefix}.txt', np.array(val_scores))
        torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')
        
