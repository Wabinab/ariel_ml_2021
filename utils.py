import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf


n_wavelengths = 55
n_timesteps = 300


class read_Ariel_dataset():
    
    def __init__(self, noisy_path, params_path, start_read):
        """
        For reading Ariel Dataset. 

        :param noisy_path: (python list of filenames) The *relative path* from the current 
            working directory to all noisy files. For local files start with "./", for 
            colab files alternatively start with "/content/" (and "./" works fine). 

        :param params_path: (python list of filenames) The *relative path* from the current
            working directory to all params files. For local files start with "./", for 
            colab files alternatively start with "/content/" (and "./" works fine). 

        :param start_read: (int)  How many data points to replace at the beginning of the 
            file. Used for preprocessing of files by replacing values before start_read
            with 1.0 to minimize impact of the drop valley. 
        """

        super().__init__()

        self.noisy_path = noisy_path
        self.params_path = params_path
        self.start_read = start_read


    def unoptimized_read_noisy(self):
        """
        Read noisy files greedily, stacking them on the first axis. 
        First axis is the time series axis. So a file with 300x55, read 
        3 files would be 900x55. 
        """
        predefined = pd.DataFrame()

        for item in self.noisy_path: 
            # Renaming the columns
            names = [item[-14:-4] + f"_{i}" for i in range(n_timesteps)]
            curr = pd.read_csv(item, delimiter="\t", skiprows=6, header=None)
            
            curr.rename(columns={x: y for x, y in zip(curr.columns, names)}, inplace=True)

            # Concatenating the pandas. 
            try: 
                predefined = pd.concat([predefined, curr], axis=1)
            except Exception as e:
                predefined = curr
                print(f"{type(e)}: {e}")

        return predefined

        

    def unoptimized_read_params(self):
        """
        Read params files greedily, stacking them on the first axis. 
        """
        raise NotImplementedError("Will be implemented in next release.")
    