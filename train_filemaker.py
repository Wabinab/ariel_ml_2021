"""
Prepare training images based on original file for use in training
"""

import numpy as np 
import os 
from utils_filemaker import ArielMLDataset
from utils import simple_transform
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Might need to change train and test path. This file currently is optimized for direct bucket
# transfer, which the utils.py is not in this case. (Not using gcp direct transfer). 

dataset = ArielMLDataset(None, params_path="./", shuffle=False, start_ind=0, max_size=125600, 
                        transform=simple_transform, device="cpu")

filenames = np.array(dataset.return_files())
loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=3)

for k, item in enumerate(tqdm(loader)):
    lc = torch.squeeze(item["lc"].detach(), 0).numpy()
    
    # Convert to image
    lc = lc.reshape((550, -1))
    assert lc.shape == (550, 300)
    
    this_filename = f'{str(filenames[k * 10]).split("/")[-1][:7]}.txt'
    
    np.savetxt(this_filename, lc)
    
    os.system(f"gsutil cp {this_filename} gs://arielml_data/training_set/aug_noisy_train/")
    os.remove(this_filename)