"""
Create check test works. 

Created on: 03 June 2021.
"""
import pytest

import os
import numpy as np
import shutil

from create_check import *

lc_train_path = "/home/chowjunwei37/ariel_ml_2021/data/noisy_train/"
params_train_path = "/home/chowjunwei37/ariel_ml_2021/data/params_train/"

@pytest.fixture
def files(tmpdir):
    return os.listdir(lc_train_path)


def setup():
    file = os.listdir(lc_train_path)
    collating(file, index_fname="indexes_test.csv", corr_fname="correlations_test.csv")


def test_indexes_csv_correct():
    assert np.array(pd.read_csv("indexes_test.csv", header=None)).shape == (3, 100)


def test_correlations_csv_correct_shape():
    assert np.array(pd.read_csv("correlations_test.csv", header=None)).shape == (3, 100)


def test_indexes_correct_file_by_type_check_int_64():
    assert type(np.array(pd.read_csv("indexes_test.csv", header=None))[0, 0]) == np.int64


def test_correlations_correct_file_by_type_check_float_64():
    assert type(np.array(pd.read_csv("correlations_test.csv", header=None))[0, 0]) == np.float64


# Ignoring test if not using kwargs to avoid accidentally overwriting files although measures are in place. 


def teardown():
    try:
        os.remove("indexes_test.csv")
    except:
        print("teardown failed")

    try:
        os.remove("correlations_test.csv")
    except:
        pass