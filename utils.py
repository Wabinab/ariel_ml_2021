"""Define generic classes and functions to facilitate baseline construction"""
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

train_mean = np.loadtxt("./data/mean.txt")  # df.mean()
train_std = np.loadtxt("./data/std.txt")  # df.std()
# train_abs = np.loadtxt("./data/mod_abs.txt")

# To not use star_logg.
train_mean = np.delete(train_mean, -5)
train_std = np.delete(train_std, -5)
# train_abs = np.delete(train_abs, -5)

class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""

    def __init__(self, lc_path, params_path=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None, mean=train_mean, 
                 std=train_std):
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
            mean: int or list or array
                mean of values
            std: int or list or array
                std dev of values
        """
        self.lc_path = lc_path
        self.transform = transform
        self.device = device
        self.mean = mean
        self.std = std

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

        # Loading extra 6 parameters. 
        # with open(item_lc_path, "r") as f:
        #     temp_storage_str = list(itertools.islice(f, 6))

        # temp_storage_float = []

        # for string in temp_storage_str:
        #     # Separate the digits and the non-digits.
        #     new_str = ["".join(x) for _, x in itertools.groupby(string, key=str.isdigit)]

        #     # Only new_str[0] is the one we want to omit.
        #     # We want to join back into a single string because "." previously is classifed
        #     # as non-digit. 
        #     new_str = "".join(new_str[1:])  

        #     # Convert to float. 
        #     temp_storage_float.append(float(new_str))

        # Transformation done here since it is easier to work with python list
        # than with numpy in terms of single element alterations, although this 
        # does introduces overheads. 
        # star temp no changes
        # star logg deleted. 
        # star rad max 1.8
        # star mass max 1.50
        # star k mag min 7.0
        # period max 16.0

        # temp_storage_float.pop(-5)
        
        # if temp_storage_float[-4] > 1.8:
        #     temp_storage_float[-4] = 1.8

        # if temp_storage_float[-3] > 1.5:
        #     temp_storage_float[-3] = 1.5

        # if temp_storage_float[-2] < 7.0:
        #     temp_storage_float[-2] = 7.0

        # if temp_storage_float[-1] > 16.0:
        #     temp_storage_float[-1] = 16.0

        lc = np.loadtxt(item_lc_path)
        # lc = lc[:, 99:200]

        # Note that the line below will automatically flatten our array
        # as it is not the same shape. This allows us to not flatten it later? 
        # I'll leave the flattening and see if it runs. If it doesn't, I'll 
        # remove the flattening (later in the code). 

        # lc = np.append(lc, temp_storage_float)

        lc = torch.from_numpy(lc)

        if self.transform:
            lc = self.transform(lc, self.mean, self.std)
        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path))
        else:
            target = torch.Tensor()
        return {'lc': lc.to(self.device),
                'target': target.to(self.device)}


def simple_transform(x, mean, std):
    """Perform a simple preprocessing of the input light curve array
    Args:
        x: np.array
            first dimension is time, at least 30 timesteps
    Return:
        preprocessed array
    """
    ##preprocessing##
    out = x.clone()


    # rand_array = np.random.rand(55 * 300 + 6)
    # centering
    # out -= rand_array
    out -= mean

    # rough rescaling
    out /= std
    # out = np.divide(out, some_std_np_array_with_same_shape)
    # out /= abs(out.mean())
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

    def __init__(self, H1=1024, H2=256, H3=256, D1=0.1, input_dim=n_wavelengths*n_timesteps, output_dim=n_wavelengths):
        """Define the baseline model for the Ariel data challenge 2021

        Args:
            H1: int
                first hidden dimension (default=1024)
            H2: int
                second hidden dimension (default=256)
            input_dim: int
                input dimension (default = 55*300+6)
            ourput_dim: int
                output dimension (default = 55)
        """
        super().__init__()
        ##model##
        # self.network = Sequential(torch.nn.Linear(input_dim, H1),
        #                            torch.nn.ReLU(),
        #                            torch.nn.Linear(H1, H2),
        #                            torch.nn.ReLU(),
        #                            torch.nn.Linear(H2, output_dim),
        #                            )
        # H1 = 256
        # H2 = 1024
        # H3 = 256
        self.network = Sequential(torch.nn.Linear(input_dim, H1),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(H1, H2),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(D1, inplace=True),
                                  torch.nn.Linear(H2, H3),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(H3, output_dim),
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