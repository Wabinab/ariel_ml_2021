"""
Define generic classes and functions to facilitate baseline construction
Some insights taken from here as well: 
https://www.kaggle.com/veb101/transfer-learning-using-efficientnet-models
"""
import copy
import itertools
import os
import numpy as np
from numpy.core.fromnumeric import transpose
import torch
from PIL import Image
import glob

from pathlib import Path
from torch.utils.data import Dataset
from torch import nn
from torch.nn import Module, Sequential
from torchvision import transforms as T

from sklearn.preprocessing import MinMaxScaler

# __author__ = "Mario Morvan"
# __email__ = "mario.morvan.18@ucl.ac.uk"


n_wavelengths = 55
n_timesteps = 300

# train_mean = np.loadtxt("./data/mean.txt")  # df.mean()
# train_std = np.loadtxt("./data/std.txt")  # df.std()
# train_abs = np.loadtxt("./data/mod_abs.txt")

# To not use star_logg.
# train_mean = np.delete(train_mean, -5)
# train_std = np.delete(train_std, -5)
# train_abs = np.delete(train_abs, -5)

class ArielMLDataset(Dataset):
    """Class for reading files for the Ariel ML data challenge 2021"""

    def __init__(self, lc_path, this_type, params_file=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None):
        """Create a pytorch dataset to read files for the Ariel ML Data challenge 2021

        Args:
            lc_path: (pathlib.Path)
                path to the folder containing the light curves files
            this_type: (str)
                "train" or "val"
            params_file: (pandas DataFrame)
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
        self.lc_path = lc_path
        self.transform = transform
        self.device = device
        self.type = this_type

        self.files = sorted(
            glob.glob(self.lc_path + "/*.png"))
        self.files = self.files[start_ind:start_ind+max_size]
        self.files = np.array(self.files)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.files)

        if params_file is not None:
            self.params_file = params_file

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item_lc_path = Path(self.lc_path) / self.files[idx]

        img = Image.open(item_lc_path)

        if self.transform:
            img = self.transform(img, self.type)

        if self.params_file is not None:
            item_params_file = self.params_file[self.files[idx].split(".")[0].split("/")[-1]]
            target = torch.from_numpy(item_params_file.to_numpy())
        else:
            target = torch.Tensor()

        return {"lc": img.float().to(self.device),
                "target": target.float().to(self.device)}

    # def __getitem__(self, idx):
    #     item_lc_path = Path(self.lc_path) / self.files[idx]

    #     lc = np.loadtxt(item_lc_path)

    #     if self.transform:
    #         lc = self.transform(lc)

    #     # lc = torch.from_numpy(lc)
        
    #     if self.params_path is not None:
    #         item_params_path = Path(self.params_path) / self.files[idx]
    #         target = torch.from_numpy(np.loadtxt(item_params_path))
    #     else:
    #         target = torch.Tensor()
    #     return {'lc': lc.to(self.device),
    #             'target': target.to(self.device)}


# def simple_transform(x, **kwargs):
#     """Perform a simple preprocessing of the input light curve array
#     Args:
#         x: np.array
#             first dimension is time, at least 30 timesteps
#     Return:
#         preprocessed array
#     """
#     ##preprocessing##
#     try:
#         out = x.clone()
#     except Exception:
#         out = x.copy()
        
#     # Min Max Scaling
#     scaler = MinMaxScaler(feature_range=(0, 255))
#     scaler.fit(out)
#     out = scaler.transform(out)

#     out = torch.from_numpy(out)

#     out = transforms.RandomResizedCrop(224)(out)
#     # out = transforms.RandomRotation(30)(out)
#     # out = transforms.RandomAffine(10, translate=(0.01, 0.12), shear=(0.01, 0.03))(out)
#     out = transforms.RandomHorizontalFlip()(out)
#     out = transforms.RandomVerticalFlip()(out)

#     assert type(out) == torch.Tensor
    
#     return out


def image_transform(x, type, **kwargs):

    # Human protein stats. 
    # mean = torch.tensor([0.05438065, 0.05291743, 0.07920227])
    # std = torch.tensor([0.39414383, 0.33547948, 0.38544176])

    # Imagenet normalization. 
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    train_trans = [
        T.RandomResizedCrop(224),
        T.RandomRotation(15),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True)
    ]

    val_trans = [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True)
    ]

    if type == "train":
        transformation = T.Compose(train_trans)
    else:
        transformation = T.Compose(val_trans)

    try:
        out = x.clone()
    except Exception:
        out = x.copy()

    out = transformation(out)

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


class TransferModel(Module):
    """
    Transfer learning model using EfficientNet-b0. Might try Resnet depending on time constraint. 
    """

    @staticmethod
    def final_layer(hidden_dim, output_dim):
        linear_layers = Sequential(nn.Linear(hidden_dim, 256),
                                   nn.LeakyReLU(0.0003),
                                   nn.Linear(256, output_dim)
        )
        return linear_layers

    def __init__(self, model_name=None, model=None, input_dim=224, output_dim=n_wavelengths):
        """
        :var model_name: (str) name of transfer learning model. 
        :var model: (model) The actual model want to transfer learn from. 
        :var input_dim: (int, tuple?) Size of image. 
        """
        super(TransferModel, self).__init__()
        self.model_name = model_name
        self.model = copy.deepcopy(model)
        # self.model = model

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = self.model._fc.in_features
        self.model._fc = TransferModel.final_layer(self.hidden_dim, self.output_dim)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        """
        Freeze unneeded training layers.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model._fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        """
        Unfreeze all layers
        """
        for param in self.model.parameters():
            param.require_grad = True

    def __repr__(self):
        return f"{self.model}"





class Baseline(Module):
    """Baseline model for Ariel ML data challenge 2021"""

    def __init__(self, H1=1024, H2=256, H3=256, input_dim=n_wavelengths*n_timesteps + 5, output_dim=n_wavelengths):
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
        self.network = Sequential(torch.nn.Linear(input_dim, H1),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(H1, H2),
                                  torch.nn.ReLU(),
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
        log_probs = torch.nn.functional.log_softmax(x, dim=1)
        # return log_probs
        return x




# # ===================================================
# # Example of transfer learning to load part of pretrained model. 
#     pretrained = torchvision.models.alexnet(pretrained=True)
#     class MyAlexNet(nn.Module):
#         def __init__(self, my_pretrained_model):
#             super(MyAlexNet, self).__init__()
#             self.pretrained = my_pretrained_model
#             self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
#                                             nn.ReLU(),
#                                             nn.Linear(100, 2))
        
#         def forward(self, x):
#             x = self.pretrained(x)
#             x = self.my_new_layers(x)
#             return x

#     my_extended_model = MyAlexNet(my_pretrained_model=pretrained)
#     my_extended_model