"""
Pytest module
Created on: 29 Avril 2021.
"""
import pytest
import os
import pandas as pd
from scipy.stats import normaltest

import warnings
warnings.simplefilter("ignore")

from test import *
from utils import *



# def test_chunks_working_correctly():
#     a = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#     assert list(chunks(a, 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


# def test_len_noisy(call_class):
#     assert call_class._noisy_len == 3


# def test_batch_list(call_class):
#     a = [1, 2, 3, 4, 5, 6]

#     assert call_class._batch_list(a, 2) == [[1, 2], [3, 4], [5, 6]]


# Tests that I want to run to make sure data is correct. 
@pytest.fixture
def call_class(tmpdir):
    return read_Ariel_dataset(noisy_path_train="./data/training_set/noisy_train",
        noisy_path_test="./data/test_set/noisy_test",
        params_path="./data/training_set/params_train", start_read=30)

# @pytest.fixture
# def call_class(tmpdir):
#     return read_Ariel_dataset(noisy_path_train="./Example_data/noisy_train",
#         noisy_path_test="./Example_data/noisy_test",
#         params_path="./Example_data/params_train", start_read=30)


# Test for all extra params in train and test are the same. 
@pytest.mark.skip
def test_all_extra_params_train_is_the_same(call_class):
    df = call_class.read_noisy_extra_param()

    # We know there are 100 permutations of BB and CC altogether. (10 BB x 10 CC)
    # So we are going to hard code. 
    df_grouped = df.filter(like='_').groupby(lambda x: x.split('_')[0], axis=1).mean()

    for AAAA in df_grouped.keys():
        pd.testing.assert_series_equal(df_grouped[AAAA], df[AAAA + "_01_01"], check_names=False)

    # Save to feature store if test passes.
    df_grouped = df.filter(like='_').groupby(lambda x: x.split('_')[0], axis=1).mean()
    df_grouped.to_csv("./data/feature_store/noisy_extra_param.csv")


@pytest.mark.skip
def test_all_extra_params_params_train_is_the_same(call_class):
    df = call_class.read_params_extra_param()

    # We know there are 100 permutations of BB and CC altogether. (10 BB x 10 CC)
    # So we are going to hard code. 
    df_grouped = df.filter(like='_').groupby(lambda x: x.split('_')[0], axis=1).mean()

    for AAAA in df_grouped.keys():
        pd.testing.assert_series_equal(df_grouped[AAAA], df[AAAA + "_01_01"], check_names=False)

    # Save to feature store if test passes.
    df_grouped = df.filter(like='_').groupby(lambda x: x.split('_')[0], axis=1).mean()
    df_grouped.to_csv("./data/feature_store/params_extra_param.csv")
