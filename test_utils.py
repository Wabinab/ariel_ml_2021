"""
Unit tests for utils.py
"""
import pytest 
import os

from utils import *


@pytest.fixture
def call_class(tmpdir):
    return read_Ariel_dataset(noisy_path="./Example_data/noisy_train",
        params_path="./Example_data/params_train", start_read=30)


# Important to run unit test correctly since it is built based on these data.
# And this architecture. 

# EXTERNAL: Test that we have the following files in our test container
# (In Github only will pass, else fail. )
def test_github_only_example_data_noisy_train_correct_items():
    our_listdir = os.listdir("./Example_data/noisy_train")

    assert "0001_01_01.txt" in our_listdir
    assert "0052_01_01.txt" in our_listdir
    assert "0100_01_01.txt" in our_listdir


def test_github_only_example_data_noisy_train_correct_length():
    our_listdir = os.listdir("./Example_data/noisy_train")

    assert len(our_listdir) == 3


# Now repeat for params
def test_github_only_example_data_params_train_correct_items():
    our_listdir = os.listdir("./Example_data/params_train")

    assert "0001_01_01.txt" in our_listdir
    assert "0052_01_01.txt" in our_listdir
    assert "0100_01_01.txt" in our_listdir


def test_github_only_example_data_params_train_correct_length()    :
    our_listdir = os.listdir("./Example_data/params_train")

    assert len(our_listdir) == 3


# Test noisy train file retrieval gets 55 rows x 900 columns
def test_noisy_train_shape(call_class):
    df = call_class.unoptimized_read_noisy()

    assert np.array(df).shape == (55, 900)


# Test names of columns have changed (i.e. for their existence.)
# And they have the correct shape (no repeats)
def test_noisy_column_names_changed(call_class):
    df = call_class.unoptimized_read_noisy()

    assert df["0100_01_01_0"].shape == (55,)
    assert df["0052_01_01_0"].shape == (55,)
    assert df["0100_01_01_0"].shape == (55,)


# Now same thing for params train, for shape changed column names. 
def test_params_train_shape(call_class):
    df = call_class.unoptimized_read_params()

    assert np.array(df).shape == (55, 3)


def test_params_column_names_changed(call_class):
    df = call_class.unoptimized_read_params()

    assert df["0100_01_01"].shape == (55,)
    assert df["0052_01_01"].shape == (55,)
    assert df["0100_01_01"].shape == (55,) 