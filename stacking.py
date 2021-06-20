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
import pandas as pd
from sklearn.metrics import mean_squared_error


project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
lc_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = project_dir / \
    "/home/chowjunwei37/Documents/data/training_set/params_train"

prefix = "stacking"

# training parameters
train_size = 120000
val_size = 5600
epochs = 15
save_from = 3

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
        torch.set_num_threads(os.cpu_count() - 1)

    error_dataset = None

    if main is True:
        looping = np.arange(4)
    else:
        looping = np.arange(4, 6)

    # First-level model training. Will train parallel so there will be another training going 
    # on at another vm, then it will upload to google cloud bucket, then main point will fetch it from
    # gcp bucket and use it. 
    for i in looping:

        if not os.path.exists(f"model/model_state_{i}.pt"):

            start = i * 50
            stop = (i + 1) * 50

            print("Training range: ", start, " ", stop)

            dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                    max_size=train_size, transform=simple_transform, device=device,
                                    start=start, stop=stop)

            dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                    max_size=val_size, transform=simple_transform, device=device,
                                    start=start, stop=stop)

            batch_size = 50

            baseline = Baseline(H1=H1, H2=H2, H3=H3, input_dim=abs(start - stop) * n_wavelengths).double().to(device)

            train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, baseline,
                                                                    epochs, save_from)

            # -------------------------------------
            # Forgot to pass "save_folder" as a str, now it is not gonna save properly
            # -------------------------------------

            np.savetxt(project_dir / f'outputs/train_losses_{i}.txt',
                np.array(train_losses))
            np.savetxt(project_dir / f'outputs/val_losses_{i}.txt', np.array(val_losses))
            np.savetxt(project_dir / f'outputs/val_scores_{i}.txt', np.array(val_scores))
            torch.save(baseline, project_dir / f'model/model_state_{i}.pt')

        else:
            print("Skipping model ", i)


    # ----------------------------------------
    # UPLOAD OR DOWNLOAD MODELS FROM GCP FOR HERE (TBC)
    if main is False:
        os.system("gsutil -m cp -r ./model/*.pt gs://arielml_data/trained_model/20062021/")
    elif main is True and len(os.listdir("model")) <= 6:
        os.system("gsutil -m cp -r gs://arielml_data/trained_model/20062021/*.pt ./model/")
    # ----------------------------------------


    if main is True and len(os.listdir("model")) >= 6:
        # First, calculate the errors on the TRAINING dataset. 
        # Ideally, they will be saved into one .csv file since we are flattening it anyways. 

        baselines = np.array([ torch.load(path_) for path_ in sorted(glob.glob("./model/*.pt")) ])

        files = sorted(
            [p for p in os.listdir(lc_train_path) if p.endswith('txt')])

        start = 0
        batch_size = 500
        torch.set_num_threads(os.cpu_count() // 2)
        torch.set_num_interop_threads(os.cpu_count() - 2)
        
        for baseline in baselines:
            train_eval_df = pd.DataFrame()

            dataset_train_eval = ArielMLDataset(lc_train_path, params_train_path, shuffle=False, transform=simple_transform,
                                    start=start, stop=50 + start)

            # If the below don't run, please SET BATCH SIZE TO 1. 
            loader_train_eval = DataLoader(dataset_train_eval, batch_size=batch_size, shuffle=False)

            for k, item in tqdm(enumerate(loader_train_eval)):  # I actually don't know why enum here
                y_true = np.array(item["target"])

                y_pred = baseline(item["lc"]).detach().numpy()  # Very inefficient way to detach all the time? 
            
                # difference error
                assert y_true.shape == (batch_size, 55)
                assert y_pred.shape == (batch_size, 55)
                abs_error = y_pred - y_true  # we are not finding the absolute error here. Just "difference". 
                
                file = str(files[k]).split(".")[0]
                train_eval_df[file] = np.array([abs_error]).flatten()
            
            start += 50

            train_eval_df.to_csv(f"./outputs/train_errors_{start}.csv", header=False, sep=",", index=False)

        # Combine csv files
        train_errors = sorted(glob.glob("./outputs/train_errors_*.csv"))
        train_eval_df = pd.DataFrame()

        for train_error in train_errors:
            train_eval_df = pd.concat([train_eval_df, train_errors], axis=0)

        assert train_eval_df.shape == (55 * 6, 125600)


        # Start preparation for training second layer model. 

        dataset_train = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=0,
                                   max_size=train_size, transform=simple_transform, device=device,
                                   error_dataset=train_eval_df)

        dataset_val = ArielMLDataset(lc_train_path, params_train_path, shuffle=True, start_ind=train_size,
                                 max_size=val_size, transform=simple_transform, device=device,
                                 error_dataset=train_eval_df)

        batch_size = 50
        torch.set_num_threads(os.cpu_count() - 1)

        total_length = (n_timesteps * n_wavelengths) + (55 * 6)  # hardcoded
        baseline = Baseline(H1=256, H2=1024, H3=1024, H4=256, input_dim=total_length, model_num=2).double().to(device)

        train_losses, val_losses, val_scores, baseline = train(batch_size, dataset_train, dataset_val, baseline,
                                                                30, save_from)

        np.savetxt(project_dir / f'outputs/train_losses_{prefix}.txt',
               np.array(train_losses))
        np.savetxt(project_dir / f'outputs/val_losses_{prefix}.txt', np.array(val_losses))
        np.savetxt(project_dir / f'outputs/val_scores_{prefix}.txt', np.array(val_scores))
        torch.save(baseline, project_dir / f'outputs/model_state_{prefix}.pt')
        
