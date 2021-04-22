import itertools
import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf


n_wavelengths = 55
n_timesteps = 300


class read_Ariel_dataset():
    
    def __init__(self, noisy_path_train, noisy_path_test, params_path, start_read):
        """
        For reading Ariel Dataset. 

        :param noisy_path_train: (str) The *relative path's parent directory* from the current 
            working directory to all noisy training files. For local files start with "./", for 
            colab files alternatively start with "/content/" (and "./" works fine). 

        :param noisy_path_train: (str) The *relative path's parent directory* from the current 
            working directory to all noisy test files. For local files start with "./", for 
            colab files alternatively start with "/content/" (and "./" works fine). 

        :param params_path: (str) The *relative path's parent directory* from the current
            working directory to all params files. For local files start with "./", for 
            colab files alternatively start with "/content/" (and "./" works fine). 

        :param start_read: (int)  How many data points to replace at the beginning of the 
            file. Used for preprocessing of files by replacing values before start_read
            with 1.0 to minimize impact of the drop valley. 
        """

        super().__init__()

        self.noisy_path = noisy_path_train
        self.noisy_path_test = noisy_path_test
        self.params_path = params_path
        self.start_read = start_read

        # list all files in path(s). 
        self.noisy_list= os.listdir(self.noisy_path)
        self.noisy_list_test = os.listdir(self.noisy_path_test)
        self.params_list = os.listdir(self.params_path)


    def _choose_train_or_test(self, folder="noisy_train"):
        """Private function to choose train or test"""

        if folder == "noisy_train":
            path = self.noisy_path
            files = self.noisy_list
        elif folder == "noisy_test":
            path = self.noisy_path_test
            files = self.noisy_list_test
        else:
            raise FileNotFoundError("Not in the list (noisy_train, noisy_test). "
                "Please input the choices in the list stated and try again.")

        return path, files


    def unoptimized_read_noisy(self, folder="noisy_train"):
        """
        Read noisy files greedily, stacking them on the first axis. 
        First axis is the time series axis. So a file with 300x55, read 
        3 files would be 900x55. 

        :param folder (str): Which folder to do baseline transition. Choices: 
            "noisy_train" (default), "noisy_test". 
        """

        path, files = self._choose_train_or_test(folder=folder)

        predefined = pd.DataFrame()

        for item in files: 
            # Concatenate filename and their parent folder. 
            relative_file_path = path + "/" + item

            # Renaming the columns
            names = [item[-14:-4] + f"_{i}" for i in range(n_timesteps)]
            curr = pd.read_csv(relative_file_path, delimiter="\t", skiprows=6, header=None)
            
            curr.rename(columns={x: y for x, y in zip(curr.columns, names)}, inplace=True)

            # Concatenating the pandas. 
            predefined = pd.concat([predefined, curr], axis=1)

        return predefined
        

    def unoptimized_read_params(self):
        """
        Read params files greedily, stacking them on the first axis. 
        """
        predefined = pd.DataFrame()

        for item in self.params_list: 
            # Relative file path: 
            relative_file_path = self.params_path + "/" + item

            names = [item[-14:-4]]  # Have to be a list to take effect
            curr = pd.read_csv(relative_file_path, delimiter="\t", skiprows=2, header=None).T

            curr.rename(columns = {x: y for x, y in zip(curr.columns, names)}, inplace=True)


            predefined = pd.concat([predefined, curr], axis=1)

        return predefined


    def read_noisy_extra_param(self):
        """
        Read the extra 6 stellar and planet parameters in noisy files. 
        """
        header = ["star_temp", "star_logg", "star_rad", "star_mass", "star_k_mag", "period"]
        
        predefined = pd.DataFrame()

        counter = 0

        for item in self.noisy_list:
            temp_storage_float = []
            relative_file_path = self.noisy_path + "/" + item

            print(relative_file_path)

            with open(relative_file_path, "r") as f:
                temp_storage_str = list(itertools.islice(f, 6))

            # Preprocess for numbers only
            for string in temp_storage_str:
                # Separate the digits and the non-digits.
                new_str = ["".join(x) for _, x in itertools.groupby(string, key=str.isdigit)]

                # Only new_str[0] is the one we want to omit.
                # We want to join back into a single string because "." previously is classifed
                # as non-digit. 
                new_str = "".join(new_str[1:])  

                # Convert to float. 
                temp_storage_float.append(float(new_str))

            # Convert to pandas DataFrame. 
            temp_storage_float = pd.DataFrame(temp_storage_float)

            # Define file name
            names = [item[-14:-4]]

            # Change the column name
            temp_storage_float.rename(columns = 
                {x: y for x, y in zip(temp_storage_float.columns, names)}, 
                inplace=True
            )

            # Change the row names for predefined (optional for readability)
            temp_storage_float.rename(index = {x: y for x, y in zip(range(6), header)},
                                    inplace=True)

            predefined = pd.concat([predefined, temp_storage_float], axis=1)

        return predefined


    def read_params_extra_param(self):
        """
        Read the extra 2 intermediate target params in the params files. 
        """
        header = ["sma", "incl"]
        
        predefined = pd.DataFrame()

        counter = 0

        for item in self.params_list:
            temp_storage_float = []
            relative_file_path = self.params_path + "/" + item

            print(relative_file_path)

            with open(relative_file_path, "r") as f:
                temp_storage_str = list(itertools.islice(f, 2))

            # Preprocess for numbers only
            for string in temp_storage_str:
                # Separate the digits and the non-digits.
                new_str = ["".join(x) for _, x in itertools.groupby(string, key=str.isdigit)]

                # Only new_str[0] is the one we want to omit.
                # We want to join back into a single string because "." previously is classifed
                # as non-digit. 
                new_str = "".join(new_str[1:])  

                # Convert to float. 
                temp_storage_float.append(float(new_str))

            # Convert to pandas DataFrame. 
            temp_storage_float = pd.DataFrame(temp_storage_float)

            # Define file name
            names = [item[-14:-4]]

            # Change the column name
            temp_storage_float.rename(columns = 
                {x: y for x, y in zip(temp_storage_float.columns, names)}, 
                inplace=True
            )

            # Change the row names for predefined (optional for readability)
            temp_storage_float.rename(index = {x: y for x, y in zip(range(6), header)},
                                    inplace=True)

            predefined = pd.concat([predefined, temp_storage_float], axis=1)

        return predefined
    

    def data_augmentation_baseline(self, folder="noisy_train"):
        """
        Data augmentation: What is being done to the data by the Baseline
            model done by the organizer. 

        :param folder (str): Which folder to do baseline transition. Choices: 
            "noisy_train" (default), "noisy_test". 
        """
        # Read file
        df = self.unoptimized_read_noisy(folder=folder)

        path, files = self._choose_train_or_test(folder=folder)

        # Transformation 1: First 30 points of each light curve are replaced
        # by 1 to reduce the impact from the ramps.

        # Get all files according to how column names are defined.
        label_names = [x[-14:-4] for x in files]

        for label_name in label_names:
            for i in range(self.start_read):
                for j in range(n_wavelengths):

                    df[str(label_name) + "_" + str(i)][j] = 1


        # Transformation 2: -1 to all data points in the file. 
        df = df - 1

        # Transformation 3: Values rescaled by dividing by 0.06 for standard deviation
        # closer to unity. 
        df /= 0.04

        return df
