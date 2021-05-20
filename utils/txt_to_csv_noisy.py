"""
File to read noisy data and params data (excluding extra parameters)
into csv files. 
Created on: 19 Mai 2021.
"""
import argparse
import io
import os
import csv
import sys
from _pytest.config import main
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Process

# from google.cloud import storage

# client = storage.Client()
# files = client.bucket("arielml_data").list_blobs(prefix="training_set/noisy_train/")

# files = list(files)

# Local file reading from Example data
noisy_path = "../Example_data/noisy_train"
params_path = "../Example_data/params_train"
noisy_test_path = "../Example_data/noisy_test"

# useful def
def return_filename_split(file):
    return [file[0:4], file[5:7], file[8:10]]

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, help="number of processes", default=os.cpu_count())
    parser.add_argument("--save_folder", type=str, help="the save location", default="./csv_files/")

    return parser.parse_args(args)

# Focusing on training files only. 
def think_of_name_later(noisy_files):
    """
    What it does is described at the top of this file. 
    
    :param save_folder: the folder in which the output csv files are to be saved. 
    """
    global noisy_path, params_path, noisy_test_path, save_folder, header

    # Read concurrent training and testing files into 2 different dataframe. 
    # header = True
    for file in tqdm(noisy_files):
        df_noisy = np.loadtxt(noisy_path + "/" + file)
        df_noisy = pd.DataFrame(df_noisy)

        assert df_noisy.shape == (55, 300)

        df_noisy.columns = df_noisy.columns.astype(str)
        
        df_params = np.loadtxt(params_path + "/" + file)
        df_params = pd.DataFrame(df_params)

        assert df_params.shape == (55, 1)

        # Rename column into "label"
        df_params.rename(columns={x: y for x, y in zip(df_params.columns, ["label"])}, inplace=True)

        # Join them into the desired shape. 
        df_joined = pd.concat([df_noisy, df_params], axis=1)

        assert df_joined.shape == (55, 301)

        df_joined = df_joined.transpose()

        # Include "primary key field" but split into 3 columns, AAAA, BB and CC. 
        df_primary_key = pd.DataFrame(return_filename_split(file)).transpose()
        df_primary_key.columns = ["AAAA", "BB", "CC"]

        assert df_primary_key.shape == (1, 3)

        # Check save_folder correct. If not, make it correct. 
        if save_folder[-1] != "/":
            save_folder += "/"

        # Create folder if not already exist. 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        for column in df_joined.columns:
            df_temp = pd.DataFrame(df_joined[column]).T

            # Drop index so concatenation happens in the correct index. 
            # Reason is because ignore_index kwargs on pd.concat does not seems to work. 
            df_temp.reset_index(drop=True, inplace=True)

            df_temp = pd.concat([df_primary_key, df_temp], axis=1)
            assert df_temp.shape == (1, 304)

            df_temp.to_csv(save_folder + f"train_table_{column}.csv", mode="a", header=header, index=False)

        # header = False

        

    print("Success")


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])

    save_folder = arguments.save_folder
    num_process = arguments.num_process

    noisy_files = os.listdir(noisy_path)

    # Pop the first data to create the first row first. 
    first_file = noisy_files.pop(0)
    first_file = [first_file]

    header = True
    think_of_name_later(first_file)

    header = False

    # Split the data into 
    noisy_files = np.array_split(noisy_files, num_process)

    with Pool(processes=num_process) as pool:

        pool.map(think_of_name_later, noisy_files)

        # think_of_name_later(noisy_path, params_path, noisy_test_path,  \
        #     noisy_files, save_folder="./csv_files/")