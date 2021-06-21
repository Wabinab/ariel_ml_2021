"""
Modified for multi-layer prediction
Created on: 21 Juin 2021.
"""
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss

# Import Dataset class 
from utils import *
from torch.utils.data.dataloader import DataLoader
    
from pathlib import Path
import tqdm
import datetime 
import os
import pandas as pd
import gc
import glob


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    n_timesteps = 300
    n_wavelengths = 55

    lc_test_path = "/home/chowjunwei37/Documents/data/test_set/noisy_test"
    
    challenge_metric = ChallengeMetric()

    # For first layer prediction. 
    for model_name in sorted(glob.glob("model/*.pt")):

        start = int(model_name[-4])

        if not os.path.exists(f"{model_name[:-3]}.csv"):

            dataset_eval = ArielMLDataset(lc_test_path, shuffle=False, transform=simple_transform, start=start * 50, stop=(start + 1) * 50)

            loader_eval = DataLoader(dataset_eval, batch_size=250, shuffle=False, num_workers=os.cpu_count() // 2)

            baseline = Baseline(input_dim=50 * n_wavelengths).double().to(device)

            baseline = torch.load(model_name)
            baseline.eval() 

            for k, item in tqdm.tqdm(enumerate(loader_eval)):

                pred = pd.DataFrame(baseline(item['lc']).detach().numpy())

                pred.to_csv(f"{model_name[:-3]}.csv", mode="a", header=None, index=False, sep="\t")

                del pred
                
                if k % 10 == 0:
                    gc.collect()

            print(f"Finish with {model_name}.")

        else:
            print(f"Skipping {model_name} as its csv already exists.")

    os.system("gsutil -m cp -r model/* gs://arielml_data/trained_model/20062021/")

    # For second layer prediction.
    test_eval_df = pd.DataFrame()
    test_errors = sorted(glob.glob("./model/*.csv"))

    for test_error in tqdm.tqdm(test_errors):
        our_df = pd.read_csv(test_error, header=None, sep="\t").T
        test_eval_df = pd.concat([test_eval_df, our_df], axis=0)

    test_eval_df.columns = sorted([p.split(".")[0] for p in os.listdir(lc_test_path) if p.endswith("txt")])

    assert test_eval_df.shape == (55 * 6, 53900)

    dataset_eval = ArielMLDataset(lc_test_path, shuffle=False, transform=simple_transform, error_dataset=test_eval_df)

    loader_eval = DataLoader(dataset_eval, batch_size=250, shuffle=False, num_workers=os.cpu_count() // 2)

    total_length = (n_timesteps * n_wavelengths) + (55 * 6)
    baseline = Baseline(input_dim=total_length, model_num=2).double().to(device)

    # Cannot use glob here because train baseline haven't fixed the train function yet. 
    # So this means there will be an extra model(s) when training for first (base) layer. 
    # So for now, we will hard code. 
    
    baseline = torch.load("outputs/model_state_stacking.pt")
    baseline.eval()

    for k, item in tqdm.tqdm(enumerate(loader_eval)):
        
        pred = pd.DataFrame(baseline(item["lc"]).detach().numpy())

        pred.to_csv(f"outputs/baseline_predict_stacking.txt", mode="a", header=None, index=False, sep="\t")

        del pred

        if k % 10 == 0:
            gc.collect()

    print(f"Finish with final prediction.")

    os.system("gsutil -m cp -r outputs/baseline_pred*.txt gs://arielml_data/trained_model/20062021/")

    


if __name__ == "__main__":
    main()