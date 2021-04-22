"""
Unit tests for utils.py
"""
import pytest 
import os

from utils import *


@pytest.fixture
def call_class(tmpdir):
    return read_Ariel_dataset(noisy_path_train="./Example_data/noisy_train",
        noisy_path_test=None,
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


# Test private choose files
def test_choose_train_or_test_correct_output(call_class):
    path, files = call_class._choose_train_or_test(folder="noisy_train")

    assert path == "./Example_data/noisy_train"


# Test noisy train file retrieval gets n_wavelengths rows x 900 columns
def test_noisy_train_shape(call_class):
    df = call_class.unoptimized_read_noisy(folder="noisy_train")

    assert np.array(df).shape == (n_wavelengths, 900)


# Test names of columns have changed (i.e. for their existence.)
# And they have the correct shape (no repeats)
def test_noisy_column_names_changed_col_name_success(call_class):
    df = call_class.unoptimized_read_noisy(folder="noisy_train")

    assert df["0100_01_01_0"].shape == (n_wavelengths,)
    assert df["0052_01_01_50"].shape == (n_wavelengths,)
    assert df["0100_01_01_100"].shape == (n_wavelengths,)


# Now same thing for params train, for shape changed column names. 
def test_params_train_shape(call_class):
    df = call_class.unoptimized_read_params()

    assert np.array(df).shape == (n_wavelengths, 3)


def test_params_column_names_changed_col_name_success(call_class):
    df = call_class.unoptimized_read_params()

    assert df["0100_01_01"].shape == (n_wavelengths,)
    assert df["0052_01_01"].shape == (n_wavelengths,)
    assert df["0100_01_01"].shape == (n_wavelengths,) 


# Check for reading extra 6 parameters in noisy file are correct ordering. 
# Each file's are appended to the columns
def test_read_noisy_extra_param_shape(call_class):
    df = call_class.read_noisy_extra_param()

    assert np.array(df).shape == (6, 3)


def test_read_noisy_extra_param_changed_col_name_success(call_class):
    df = call_class.read_noisy_extra_param()

    assert df["0100_01_01"].shape == (6,)
    assert df["0052_01_01"].shape == (6,)
    assert df["0100_01_01"].shape == (6,)


def test_read_noisy_extra_param_data(call_class):
    df = call_class.read_noisy_extra_param()

    for i in range(6):
        assert type(df["0100_01_01"][i]) == np.float64


# Same test for params 2 parameters: semimajor axis (sma) and inclination (incl) test. 
def test_read_params_extra_param_shape(call_class):
    df = call_class.read_params_extra_param()

    assert np.array(df).shape == (2, 3)


def test_read_params_extra_param_changed_col_name_success(call_class):
    df = call_class.read_params_extra_param()

    assert df["0100_01_01"].shape == (2,)
    assert df["0052_01_01"].shape == (2,)
    assert df["0001_01_01"].shape == (2,)


def test_read_params_extra_param_data(call_class):
    df = call_class.read_params_extra_param()

    for i in range(2):
        assert type(df["0052_01_01"][i]) == np.float64


# Check that Baseline model transformation works as expected. 
def test_data_augmentation_baseline_replace_first_N_points(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_train")

    items = ["0001_01_01", "0052_01_01", "0100_01_01"]

    test_names = [item + f"_{i}" for item in items for i in range(30)]

    aggregate = 0

    for i in test_names:
        for j in range(n_wavelengths):
            aggregate += df[i][j]

    result = aggregate / (len(test_names) * n_wavelengths)

    assert result == pytest.approx(1.00, 0.01)