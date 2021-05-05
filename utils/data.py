"""
File for performing Data Transformation across all files and save it into a new file. 
Created on: 28 Avril 2021.
"""
import matplotlib.pyplot as plt
import numpy as np
import os 
import modin.pandas as pd 
from pathlib import Path
from utils import read_Ariel_dataset

import warnings
warnings.simplefilter("ignore")

from distributed import Client

client = Client()


class Transform_Data(read_Ariel_dataset):
    
    def __init__(self):
        super().__init__()

    def transform_steps():
        """
        Read from text file, 
        Read the extra params from text file
        Do the transformation
        Write the extra params into text file
        Write the transformed time series into text file. 
        """

        raise NotImplementedError("Yet to be implemented")
    