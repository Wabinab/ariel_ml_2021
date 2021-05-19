"""
File to read noisy data and params data (excluding extra parameters)
into csv files. 
Created on: 19 Mai 2021.
"""
import io
import os
import csv
from _pytest.config import main
import numpy as np
import pandas as pd

# from google.cloud import storage

# client = storage.Client()
# files = client.bucket("arielml_data").list_blobs(prefix="training_set/noisy_train/")

# files = list(files)

# Local file reading from Example data
noisy_path = "../Example_data/noisy_train"
params_path = "../Example_data/params_train"
noisy_test_path = "../Example_data/noisy_test"

# Focusing on training files only. 
def think_of_name_later(noisy_path, params_path, noisy_test_path, save_folder):
    """
    What it does is described at the top of this file. 
    
    :param save_folder: the folder in which the output csv files are to be saved. 
    """
    noisy_files = os.listdir(noisy_path)
    params_files = os.listdir(params_path)

    # Read concurrent training and testing files into 2 different dataframe. 
    header = True
    for file in noisy_files:
        df_noisy = np.loadtxt(noisy_path + "/" + file)
        df_noisy = pd.DataFrame(df_noisy)

        assert df_noisy.shape == (55, 300)

        #str_column_names = df_noisy.columns.astype(str)
        #df_noisy.rename(columns={x: y for x, y in zip(df_noisy.columns, str_column_names)}, inplace=True)

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

        # Check save_folder correct. If not, make it correct. 
        if save_folder[-1] != "/":
            save_folder += "/"

        # Create folder if not already exist. 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # header = True
        # for column in df_joined.columns:
        #     print(pd.DataFrame(df_joined[column]).T)
        #     break
        # break

        
        for column in df_joined.columns:
            df_temp = pd.DataFrame(df_joined[column]).T
            df_temp.to_csv(save_folder + f"train_table_{column}.csv", mode="a", header=header, index=False)

        header = False

        

        # Split the rows to store in 55 different csv. 
        # header = True
        # for index, row in df_joined.iterrows():
        #     temp_row = pd.DataFrame(row.values.reshape(1, 301))
        #     temp_row.to_csv(save_folder + f"train_table_{index}.csv", mode="a", header=header)
        #     header = False



        


    print("Success")


if __name__ == "__main__":
    think_of_name_later(noisy_path, params_path, noisy_test_path, save_folder="./csv_files/")