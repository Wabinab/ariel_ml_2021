import itertools
import os
import numpy as np
from numpy.core.fromnumeric import transpose
import torch
from PIL import Image

from pathlib import Path
from torch.utils.data import Dataset
from torch import nn
from torch.nn import Module, Sequential
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler

n_wavelengths = 55
n_timesteps = 300

class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""

    def __init__(self, lc_path, params_path=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None, transpose=None,
                 skip_cc=False):
        """Create a pytorch dataset to read files for the Ariel ML Data challenge 2021
        Args:
            lc_path: str
                path to the folder containing the light curves files
            params_path: str
                path to the folder containing the target transit depths (optional)
            transform: callable
                transformation to apply to the input light curves
            start_ind: int
                where to start reading the files from (after ordering)
            max_size: int
                maximum dataset size
            shuffle: bool
                whether to shuffle the dataset order or not
            seed: int
                numpy seed to set in case of shuffling
            device: str
                torch device
        """
        self.lc_path = Path(lc_path)
        self.transform = transform
        self.transpose = transpose
        self.device = device

        if skip_cc:
            self.files = sorted(
                [p for p in self.lc_path.glob("*_01.txt")])  
        else:
            self.files = sorted(
                [p for p in self.lc_path.iterdir() if p.suffix == "txt"]
            )
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.files)
        self.files = self.files[start_ind:start_ind+max_size]

        if params_path is not None:
            self.params_path = params_path
        else:
            self.params_path = None
            self.params_files = None

    def __len__(self):
        return len(self.files)

    def return_files(self):
        return self.files

    def __getitem__(self, idx):
        item_lc_path = Path(self.lc_path) / self.files[idx]

        lc = np.loadtxt(item_lc_path)

        lc = torch.from_numpy(lc)

        if self.transform:
            lc = self.transform(lc)

        if self.transpose is True: 
            lc = lc.T
        
        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path))
        else:
            target = torch.Tensor()
        return {'lc': lc.to(self.device),
                'target': target.to(self.device)}