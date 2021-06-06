import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
# import pathlib

# project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs

def collating(files, **kwargs):
    """
    Collect the 101 (including self) most important attributes (excluding extra params)
    from each and every single file and input it into bigquery. 

    This will not state which file is which and will treat, all else being equal, a 
    blackbox. 

    Note since we are using multiprocessing, there are no confirmation they are ordered
    in the way they should be. Nor do we need them to be ordered. 

    We only want to see how large a correlation could be, and most importantly, what
    columns are most important to all the files (comes out the most). 
    """ 
    try:
        index_fname = kwargs["index_fname"]

        if index_fname[-4] != ".":
            index_fname += ".csv"
    except KeyError:
        index_fname = "indexes.csv"

    try:
        corr_fname = kwargs["corr_fname"]

        if corr_fname[-4] != ".":
            index_fname += ".csv"
    except KeyError:
        corr_fname = "correlations.csv"


    for file in tqdm(files):
        df = pd.read_csv(lc_train_path + file, delimiter="\t", header=None, skiprows=6)
        target = pd.read_csv(params_train_path + file, delimiter="\t", header=None, skiprows=2)

        target = target.T
        target.columns = ["label"]

        total = pd.concat([df, target], axis=1)

        corr = total.corr()["label"].abs().nlargest(101)
        _ = corr.pop("label")

        # Save indexes (the columns that are most important)
        index = pd.DataFrame(np.array(corr.index).reshape((1, 100)), dtype="int")

        # Save correlation values (their corresponding correlation strengths)
        correlations = pd.DataFrame(corr.values.reshape((1, 100)), dtype="float")

        index.to_csv(index_fname, sep=",", header=False, mode="a", index=False)

        correlations.to_csv(corr_fname, sep=",", header=False, mode="a", index=False)
        
        
        # with open(index_fname, "a") as f:
        #     np.savetxt(f, index.reshape((1, 100)), delimiter=",")


        # # Save values (their corresponding correlation strengths)
        # with open("correlation.csv", "a") as f:
        #     np.savetxt(f, corr.values.reshape((1, 100)), delimiter=",")


if __name__ == "__main__":
    lc_train_path = "/home/chowjunwei37/Documents/data/training_set/noisy_train/"
    params_train_path = "/home/chowjunwei37/Documents/data/training_set/params_train/"
    
    num_processes = os.cpu_count()
    noisy_files = os.listdir(lc_train_path)
    noisy_files = np.array_split(noisy_files, num_processes)

    with Pool(processes=num_processes) as pool:
        pool.map(collating, noisy_files)

    os.rename("indexes.csv", "./outputs/indexes.csv")
    os.rename("correlations.csv", "./outputs/correlations.csv")
