import copy
import itertools
import os
import numpy as np
import modin.pandas as pd
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from scipy.stats import yeojohnson
from tqdm import tqdm

import tensorflow as tf

import warnings
warnings.simplefilter("ignore")

from distributed import Client

client = Client()


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

        # Grouped by AAAA: 
        self.group_noisy_list = self._group_list(self.noisy_list)
        self.group_noisy_list_test = self._group_list(self.noisy_list_test)
        self.group_params_list = self._group_list(self.params_list)


    def _group_list_return(self):
        """
        Only used for unit test purposes. 

        Return self.group_noisy_list and assert it is true. 
        """
        return self.group_noisy_list


    def _choose_train_or_test(self, folder="noisy_train", batch_size=1):
        """Private function to choose train or test. 
        
        :param batch_size (int): The batch size to take. NotImplemented yet. 
        """

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


    def _len_noisy_list(self):
        return len(self.noisy_list)


    def unoptimized_read_noisy(self, folder="noisy_train", **kwargs):
        """
        Read noisy files greedily, stacking them on the first axis. 
        First axis is the time series axis. So a file with 300x55, read 
        3 files would be 900x55. 
        :param folder (str): Which folder to do baseline transition. Choices: 
            "noisy_train" (default), "noisy_test". 
        """

        path, files = self._choose_train_or_test(folder=folder, **kwargs)

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


    def _group_list(self, mylist):
        """
        Group list together. Here the function is specific to group AAAA together into 
        a sublist to not cramp the memory and dataframe I/O.
        """
        return [list(v) for i, v in itertools.groupby(mylist, lambda x: x[:4])]


    def read_noisy_extra_param(self, folder="train", saveto="./feature_store/noisy_train"):
        """
        Read the extra 6 stellar and planet parameters in noisy files. 

        :param folder (str): "train" or "test" choice. Default "train" for noisy train set. 
        :param saveto (str): The directory to save to. Will make the directory if not 
            already exists.
        """
        header = ["star_temp", "star_logg", "star_rad", "star_mass", "star_k_mag", "period"]
        
        predefined = pd.DataFrame()

        if saveto[-1] != "/":
            saveto += "/"

        try:
            os.makedirs(saveto)
        except OSError as e:
            pass

        if folder == "train":
            path = self.noisy_path
            mylist = self.group_noisy_list
        elif folder == "test":
            path = self.noisy_path_test
            mylist = self.group_noisy_list_test
        else: 
            raise ValueError("Invalid 'folder' entry. Please choose between 'train' or 'test'.")

        # To ensure small enough, read them into groups of csv first. 
        for grouped_item in tqdm(mylist):

            for item in grouped_item:
                temp_storage_float = []
                relative_file_path = path + "/" + item

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

            predefined.to_csv(saveto + item[:4] + ".csv")
            
            # Reset predefined
            predefined = pd.DataFrame()

        # Then concatenate the csv files. 
        saved_list = os.listdir(saveto)
        predefined = pd.DataFrame()

        for item in saved_list:
            relative_file_path = saveto + item

            name = [item[:-4]]  # ignore the .csv at the end. 

            temp_df = pd.read_csv(relative_file_path, index_col=0)

            predefined = pd.concat([predefined, temp_df], axis=1)

        return predefined


    def read_params_extra_param(self, saveto="./feature_store/params_train"):
        """
        Read the extra 2 intermediate target params in the params files. 
        """
        header = ["sma", "incl"]
        
        predefined = pd.DataFrame()

        if saveto[-1] != "/":
            saveto += "/"

        try:
            os.makedirs(saveto)
        except OSError as e:
            pass

        mylist = self.group_params_list  # Since we only have one folder, so hardcoded here. 
       
        for grouped_item in tqdm(mylist):

            for item in grouped_item:
                temp_storage_float = []
                relative_file_path = self.params_path + "/" + item

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

            predefined.to_csv(saveto + item[:4] + ".csv")

            # Reset predefined
            predefined = pd.DataFrame()

        # Then concatenate the csv files. 
        saved_list = os.listdir(saveto)
        predefined = pd.DataFrame()

        print(saved_list)

        for item in saved_list:
            relative_file_path = saveto + item

            name = [item[:-4]]  # ignore the .csv at the end. 

            temp_df = pd.read_csv(relative_file_path, index_col=0)

            predefined = pd.concat([predefined, temp_df], axis=1)

        return predefined
    

    def data_augmentation_baseline(self, folder="noisy_train", extra_transform=None, **kwargs):
        """
        Data augmentation: What is being done to the data by the Baseline
            model done by the organizer. 
        :param folder (str): Which folder to do baseline transition. Choices: 
            "noisy_train" (default), "noisy_test". 
        :param extra_transform (str): Are there any other transformation you would like
            to make before going into final transform? Note: only restricted support. 
            Choose from "log", "sqrt" and "square". 
        """
        # Read file
        df = self.unoptimized_read_noisy(folder=folder, **kwargs)

        path, files = self._choose_train_or_test(folder=folder, **kwargs)

        # Transformation 1: First 30 points of each light curve are replaced
        # by 1 to reduce the impact from the ramps.

        # Get all files according to how column names are defined.
        label_names = [x[-14:-4] for x in files]

        for label_name in label_names:
            for i in range(self.start_read):
                for j in range(n_wavelengths):

                    df[str(label_name) + "_" + str(i)][j] = 1


        # Extra transformation outside of what is being done in baseline. 
        # Tests yet to be implemented. 
        for i in range(n_wavelengths):
            
            if extra_transform == "log":
                df.iloc[i] = np.log(df.iloc[i])

            elif extra_transform == "sqrt":
                df.iloc[i] = np.sqrt(df.iloc[i])

            elif extra_transform == "square":
                df.iloc[i] = np.square(df.iloc[i])


        # Transformation 2: -1 to all data points in the file. 
        df = df - 1

        # Transformation 3: Values rescaled by dividing by 0.06 for standard deviation
        # closer to unity. 
        df /= 0.04

        return df


    def read_noisy_vstacked(self, from_baseline=True, dataframe=None, **kwargs):
        """
        Read file vstacked on each other instead of concatenating along the column. 
        So for example, our file with timestep of 300 for 3 files, instead of returning
        for one single wavelength shape of (1, 900) will return (3, 300) instead. 
        This way we aggregate all one single wavelength onto one block and continue vstacking
        downwards, keeping the rows = 300 constant. 
        :param from_baseline (bool): get data from data_augmentation_baseline
            directly or insert data yourself? Default to True. 
        
        :param dataframe (pandas.DataFrame): the data to be passed in. Only to be used
            if from_baseline = False, otherwise default to None.
        """
        if from_baseline == True:
            df = self.unoptimized_read_noisy(**kwargs)

        else:
            df = dataframe

        new_df = pd.DataFrame()

        for key, value in df.iterrows():

            start_count_sectors = 0
            end_count_sectors = n_timesteps

            # To iterate for every 300 timesteps since this is from a single file. 
            while end_count_sectors <= len(value):

                data = np.array(value[start_count_sectors: end_count_sectors])

                new_df = new_df.append(pd.DataFrame(data).T, ignore_index = True)

                start_count_sectors = end_count_sectors
                end_count_sectors += n_timesteps

        return new_df



    def yeo_johnson_transform(self, from_baseline=True, dataframe=None, original_shape=True, **kwargs):
        """
        The Yeo-Johnson Transform: https://www.stat.umn.edu/arc/yjpower.pdf
        To "normalize" a non-normal distribution (i.e. transform from non-Gaussian
        to Gaussian distribution), for a mix of positive and negative numbers, 
        (or strictly positive or strictly negative). 
        :param from_baseline (bool): get data from data_augmentation_baseline
            directly or insert data yourself? Default to True. 
        
        :param dataframe (pandas.DataFrame): the data to be passed in. Only to be used
            if from_baseline = False, otherwise default to None.
        :param original_shape (bool): Whether to concatenate back to original shape of (x, 55). 
            If not True, it will choose a shape of (300, y) instead for easy reading. 
            Defaults to True. 
        """
        if from_baseline == True:
            df = self.data_augmentation_baseline(**kwargs)

        else:
            df = dataframe

        # pt = PowerTransformer(method="yeo-johnson")

        try:
            new_df = pd.DataFrame()

            for key, value in df.iterrows():
                
                temp_array = []

                start_count_sectors = 0
                end_count_sectors = n_timesteps

                # To iterate for every 300 timesteps since this is from a single file. 
                while end_count_sectors <= len(value):

                    data = np.array(value[start_count_sectors: end_count_sectors])

                    # # Manual method instead of using built-in library in scipy. 
                    # data = data.reshape(-1, 1)
                    # pt.fit(data)
                    # transformed_data = pt.transform(data)

                    transformed_data, _ = yeojohnson(data)

                    if original_shape == True:
                        temp_array += list(transformed_data)
                    else:
                        new_df = new_df.append(pd.DataFrame(transformed_data).T, ignore_index = True)

                    start_count_sectors = end_count_sectors
                    end_count_sectors += n_timesteps

                if original_shape == True:
                    new_df = new_df.append(pd.DataFrame(temp_array).T, ignore_index = True)

        except AttributeError as e:
            # 'Series' object has no attribute 'iterrows'

            data = np.array(df)
            data = data.reshape(-1, 1)
            pt.fit(data)
            transformed_data = pt.transform(data)

            new_df = transformed_data

        return new_df


    def flow_from_directory():
        """
        Flow directly from directory with batch size = 1. 
        """
        raise NotImplementedError("Yet to be implemented.")