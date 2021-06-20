"""Define generic classes and functions to facilitate baseline construction"""
import copy
import itertools
import os
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import Module, Sequential

# __author__ = "Mario Morvan"
# __email__ = "mario.morvan.18@ucl.ac.uk"


n_wavelengths = 55
n_timesteps = 300

class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""

    def __init__(self, lc_path, params_path=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None, start=None, 
                 stop=None, error_dataset=None):
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
            error_dataset: pd.DataFrame
                The error (loss) with the same shape as the training data. (normalized)
        """
        self.lc_path = lc_path
        self.transform = transform
        self.device = device
        self.start = start
        self.stop = stop
        self.error_dataset = error_dataset

        self.files = sorted(
            [p for p in os.listdir(self.lc_path) if p.endswith('txt')])
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

    def __getitem__(self, idx):
        item_lc_path = Path(self.lc_path) / self.files[idx]

        lc = np.loadtxt(item_lc_path)
        lc = lc[:, self.start:self.stop]

        if self.transform:  # Transform first so we don't have to worry about how to do for error.
            lc = self.transform(lc)

        file_without_dottxt = None

        if self.error_dataset is not None:
            file_without_dottxt = str(self.files[idx]).split(".")[0]
            error_numpy = np.array( self.error_dataset[file_without_dottxt] )
            lc = np.append(lc, error_numpy)

        lc = torch.from_numpy(lc)  

        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path))
        else:
            target = torch.Tensor()
        return {'lc': lc.to(self.device),
                'target': target.to(self.device)}


def simple_transform(x):
    """Perform a simple preprocessing of the input light curve array
    Args:
        x: np.array
            first dimension is time, at least 30 timesteps
    Return:
        preprocessed array
    """
    ##preprocessing##
    try:
        out = x.clone()
    except Exception:
        out = copy.deepcopy(x)  # Is this memory efficient? 


    # rand_array = np.random.rand(55 * 300)
    # centering
    out -= 1

    # rough rescaling
    out /= 0.04
    return out


class ChallengeMetric:
    """Class for challenge metric"""

    def __init__(self, weights=None):
        """Create a callable object close to the Challenge's metric score

        __call__ method returns the error and score method returns the unweighted challenge metric

        Args:
            weights: iterable
                iterable containing the weights for each observation point (default None will create unity weights)
        """
        self.weights = weights

    def __call__(self, y, pred):
        """Return the unweighted error related to the challenge, as defined (here)[https://www.ariel-datachallenge.space/ML/documentation/scoring]

        Args:
            y: torch.Tensor
                target tensor
            pred: torch.Tensor
                prediction tensor, same shape as y
        Return: torch.tensor
            error tensor (itemisable), min value = 0
        """
        y = y
        pred = pred
        if self.weights is None:
            weights = torch.ones_like(y, requires_grad=False)
        else:
            weights = self.weights

        return (weights * y * torch.abs(pred - y)).sum() / weights.sum() * 1e6

    def score(self, y, pred):
        """Return the unweighted score related to the challenge, as defined (here)[https://www.ariel-datachallenge.space/ML/documentation/scoring]

        Args:
            y: torch.Tensor
                target tensor
            pred: torch.Tensor
                prediction tensor, same shape as y
        Return: torch.tensor
            score tensor (itemisable), max value = 10000
        """
        y = y
        pred = pred
        if self.weights is None:
            weights = torch.ones_like(y, requires_grad=False)
        else:
            weights = self.weights

        return (1e4 - 2 * (weights * y * torch.abs(pred - y)).sum() / weights.sum() * 1e6)


class Baseline(Module):
    """Baseline model for Ariel ML data challenge 2021"""

    def __init__(self, H1=1024, H2=256, H3=256, H4=256, model_num=1,
                 input_dim=n_wavelengths*n_timesteps, output_dim=n_wavelengths):
        """Define the baseline model for the Ariel data challenge 2021

        Args:
            H1: int
                first hidden dimension (default=1024)
            H2: int
                second hidden dimension (default=256)
            input_dim: int
                input dimension (default = 55*300)
            ourput_dim: int
                output dimension (default = 55)
        """
        super().__init__()

        if model_num == 1:
            self.network = Sequential(torch.nn.Linear(input_dim, H1),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H1, H2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H2, H3),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H3, output_dim),
                                    )  

        else:   
            # Not sure if this will overfit since we don't regularize it. 
            self.network = Sequential(torch.nn.Linear(input_dim, H1),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H1, H2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H2, H3),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H3, H4),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(H4, output_dim),
                                    ) 
        


    def __call__(self, x):
        """Predict rp/rs from input tensor light curve x"""
        out = torch.flatten(
            x, start_dim=1)  # Need to flatten out the input light curves for this type network
        out = self.network(out)
        return out


 ## convolution nn ##
class BaselineConv(Module):

    def __init__(self, H1=1024, H2 = 256, input_dim=n_wavelengths*n_timesteps + 5, output_dim=n_wavelengths):
        super().__init__()
        
        self.layer1 = torch.nn.Conv1d(in_channels=input_dim, out_channels=H1, kernel_size=1, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=H1, out_channels=H2, kernel_size=1)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(H2, output_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        # x = torch.reshape(x, (100 ,3301, 5))  # batch_size, input_dim // kernel_size, kernel_size

        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer3(x)
        # log_probs = torch.nn.functional.log_softmax(x, dim=1)
        # return log_probs
        return x