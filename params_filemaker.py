"""
Prepare params files for use in training
"""
import numpy as np 
import os
import pandas as pd 
from pathlib import Path
from tqdm import tqdm
from utils_filemaker import ArielMLDataset
import torch
from torch.utils.data import DataLoader

# Requires another utils.py file found on github on branch dev_11062021

lc_train_path = "/home/chowjunwei37/Documents/data/training_set/noisy_train"
params_train_path = "/home/chowjunwei37/Documents/data/training_set/params_train"

device = torch.device("cpu")

dataset = ArielMLDataset(lc_train_path, params_path=params_train_path, 
                        shuffle=False, start_ind=0, max_size=125600, 
                        device=device, skip_cc=True)

filenames = np.array(dataset.return_files())
# For batch size larger than one, requires modification to this file. 
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() // 2)

filenames = np.array(dataset.return_files())
df = pd.DataFrame()

for k, item in enumerate(tqdm(loader)):
    target = torch.squeeze(item["target"].detach(), 0).numpy()

    assert target.shape == (55, )

    # Append to pandas dataframe
    df_columns = df.columns
    df = pd.concat([df, pd.DataFrame(target)], axis=1)
    df.columns = np.append(df_columns.values, str(filenames[k]).split("/")[-1][:7])

    # Note here have no failsave. THis should run to the end or you lose everything. 

df.to_csv("outputs/params_aug.csv", index=False, header=True)
os.system("gsutil cp outputs/params_aug.csv gs://arielml_data/training_set/")