"""
Unit tests for utils.py
"""
import pytest 
import os
from scipy.stats import normaltest

import warnings
warnings.simplefilter("ignore")

from utils import *


@pytest.fixture
def call_class(tmpdir):
    return read_Ariel_dataset(noisy_path_train="./Example_data/noisy_train",
        noisy_path_test="./Example_data/noisy_test",
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


def test_github_only_example_data_params_train_correct_length():
    our_listdir = os.listdir("./Example_data/params_train")

    assert len(our_listdir) == 3


def test_github_only_example_data_noisy_test_correct_items():
    our_listdir = os.listdir("./Example_data/noisy_test")

    assert "0012_05_05.txt" in our_listdir
    assert "0072_05_05.txt" in our_listdir
    assert "0205_05_05.txt" in our_listdir


def test_github_only_example_data_noisy_test_correct_length():
    our_listdir = os.listdir("./Example_data/noisy_test")

    assert len(our_listdir) == 3


# Test private choose files
def test_choose_train_or_test_correct_output_train(call_class):
    path, files = call_class._choose_train_or_test(folder="noisy_train", batch_size=3)

    assert path == "./Example_data/noisy_train"


def test_choose_train_or_test_correct_output_train_files(call_class):
    # Assumed you have pass the test where only 3 files in the directory. 

    path, files = call_class._choose_train_or_test(folder="noisy_train", batch_size=3)

    assert len(files) == 3


def test_len_noisy_list(call_class):
    assert call_class._len_noisy_list() == 3


def test_len_noisy_list_after_call_once_batch_size_one(call_class):
    path, files = call_class._choose_train_or_test(folder="noisy_train", batch_size=1)

    assert call_class._len_noisy_list() == 2


def test_choose_train_or_test_correct_output_test(call_class):
    path, files = call_class._choose_train_or_test(folder="noisy_test")

    assert path == "./Example_data/noisy_test"


def test_choose_train_or_test_correct_output_test_files(call_class):
    # Assumed you have pass the test where only 3 files in the directory. 

    path, files = call_class._choose_train_or_test(folder="noisy_test", batch_size=3)

    assert len(files) == 3


def test_choose_train_or_test_correct_output_error(call_class):
    with pytest.raises(FileNotFoundError) as e:
        path, files = call_class._choose_train_or_test(folder="raise an error")


# Test noisy train file retrieval gets n_wavelengths rows x 900 columns
def test_noisy_train_shape(call_class):
    df = call_class.unoptimized_read_noisy(folder="noisy_train", batch_size=3)

    assert np.array(df).shape == (n_wavelengths, 900)


# Test names of columns have changed (i.e. for their existence.)
# And they have the correct shape (no repeats)
def test_noisy_column_names_changed_col_name_success(call_class):
    df = call_class.unoptimized_read_noisy(folder="noisy_train", batch_size=3)

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
# To be changed when transformation changes. 
@pytest.mark.slow
def test_data_augmentation_baseline_replace_first_N_points(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_train", batch_size=3)

    items = ["0001_01_01", "0052_01_01", "0100_01_01"]

    test_names = [item + f"_{i}" for item in items for i in range(30)]

    aggregate = 0

    for i in test_names:
        for j in range(n_wavelengths):
            aggregate += df[i][j]

    result = aggregate / (len(test_names) * n_wavelengths)

    assert result == pytest.approx(0.00, 0.001)


# To be changed when transformation changes. 
@pytest.mark.slow
def test_data_augmentation_baseline_no_replace_other_points(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_train", batch_size=3)

    max_add = 0
    min_add = 0

    for i in range(n_wavelengths):
        max_add += df.max(axis=1)[i]
        min_add += df.min(axis=1)[i]

    max_add /= n_wavelengths
    min_add /= n_wavelengths

    assert max_add > 0.00
    assert min_add < 0.00


@pytest.mark.skip("No proper implementation for this.")
def test_data_augmentation_baseline_mean_is_near_zero(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_train", batch_size=3)

    assert type(df.mean(1).min()) == np.float64

    assert df.mean(1).min() == pytest.approx(0.1, 0.15)
    assert df.mean(1).max() == pytest.approx(0.1, 0.15)



@pytest.mark.skip("No proper implementation for this. ")
def test_data_augmentation_baseline_standard_deviation_near_one(call_class):
    # Here we assume mean is 0.00 if we passed the test above. 
    df = call_class.data_augmentation_baseline(folder="noisy_train", batch_size=3)

    assert df.std(1).max() > 1
    assert df.std(1).min() < 1


# Yeo-Johnson transformation
@pytest.mark.skip("We do not have a Gaussian Graph so even the transformation it doesn't work.")
def test_yeo_johnson_transform_normal_dist_threshold_one_thousandths(call_class):
    df = call_class.yeo_johnson_transform(folder="noisy_train", batch_size=3)
    
    for key, data in df.iterrows():
        k2, p = normaltest(data)

        assert p <= 0.001


@pytest.mark.skip("We do not have a Gaussian Graph so even the transformation it doesn't work.")
def test_without_yeo_johnson_transform_not_normal(call_class):
    df = call_class.data_augmentation_baseline(batch_size=3)
    ptot = 0

    for key, data in df.iterrows():
        k2, p = normaltest(data)

        ptot += p

    assert ptot >= 0.001


@pytest.mark.skip("We do not have a Gaussian Graph so even the transformation it doesn't work.")
def test_yeo_johnson_transform_original_shape_not_true_pass_p_test(call_class):
    df = call_class.yeo_johnson_transform(folder="noisy_train", original_shape=False, batch_size=3)

    for key, data in df.iterrows():
        k2, p = normaltest(data)

        assert p <= 0.001


@pytest.mark.slow
def test_yeo_johnson_transform_original_shape_not_true_correct_shape(call_class):
    df = call_class.yeo_johnson_transform(folder="noisy_train", original_shape=False, batch_size=3)

    assert df.shape == (165, 300)


@pytest.mark.slow
def test_yeo_johnson_transform_original_shape_true_correct_shape(call_class):
    df = call_class.yeo_johnson_transform(folder="noisy_train", batch_size=3)

    assert df.shape == (55, 900)


@pytest.mark.slow
def test_yeo_johnson_transform_pass_in_alternative_df(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_test", batch_size=3)

    df = call_class.yeo_johnson_transform(from_baseline=False, dataframe=df)

    assert df.shape == (55, 900)


@pytest.mark.slow
def test_read_noisy_vstacked_correct_shape(call_class):
    df = call_class.read_noisy_vstacked(folder="noisy_train", batch_size=3)

    assert df.shape == (165, 300)


@pytest.mark.slow
def test_read_noisy_vstacked_pass_in_alternative_df(call_class):
    df = call_class.data_augmentation_baseline(folder="noisy_test", batch_size=3)

    df = call_class.read_noisy_vstacked(from_baseline=False, dataframe=df)

    assert df.shape == (165, 300)
