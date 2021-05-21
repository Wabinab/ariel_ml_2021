"""
Unit Test for txt_to_csv_noisy.py
"""
import pytest
import shutil
import os
import pandas as pd
from txt_to_csv_noisy import *

noisy_path = "./Example_data/noisy_train"
params_path = "./Example_data/params_train"
noisy_test_path = "./Example_data/noisy_test"
save_folder = "./test_temp/"
header = True

def test_return_filename_split_works_as_expected():
    assert return_filename_split("0101_02_03.txt") == ["0101", "02", "03"]


def test_parse_arguments_work_correctly():
    parsed = parse_arguments(["--num_process", "2", "--save_folder", "./csv_files", 
        "--noisy_path", noisy_path, "--params_path", params_path, "--noisy_test_path", noisy_test_path])

    assert parsed.num_process == 2
    assert parsed.save_folder == "./csv_files"
    assert parsed.noisy_path == noisy_path
    assert parsed.params_path == params_path
    assert parsed.noisy_test_path == noisy_test_path


@pytest.mark.skip
def test_parse_arguments_can_be_any_order():
    parsed = parse_arguments(["--save_folder", "./csv_files", "--num_process", "2"])

    assert parsed.save_folder == "./csv_files"
    assert parsed.num_process == "2"


def test_parse_arguments_default_work_correctly():
    parsed = parse_arguments([])

    assert parsed.num_process == os.cpu_count()
    assert parsed.save_folder == "./csv_files/"
    assert parsed.noisy_path == "./data/training_set/noisy_train/"
    assert parsed.noisy_test_path == "./data/test_set/noisy_test/"
    assert parsed.params_path == "./data/training_set/params_train/"


# Build-up for next few tests. 
def buildup_training_files_to_csv():
    # Local file reading from Example data
    noisy_path = "./Example_data/noisy_train/"
    params_path = "./Example_data/params_train/"
    noisy_test_path = "./Example_data/noisy_test/"
    save_folder = "./test_temp/"
    header = True
    num_process = 2

    noisy_files = os.listdir(noisy_path)
    first_file = noisy_files.pop(0)
    first_file = [first_file]

    training_files_to_csv(first_file, noisy_path=noisy_path, params_path=params_path,
        noisy_test_path=noisy_test_path, save_folder=save_folder, header=header)

    header = False

    training_files_to_csv(noisy_files, noisy_path=noisy_path, params_path=params_path,
        noisy_test_path=noisy_test_path, save_folder=save_folder, header=header)


@pytest.mark.slow
def test_training_files_to_csv_presence_of_folder():
    buildup_training_files_to_csv()

    assert os.path.isdir("./test_temp/")

    my_teardown()


@pytest.mark.slow
def test_training_files_to_csv_presence_of_all_csv_files():
    buildup_training_files_to_csv()

    what_indir = os.listdir("./test_temp/")
    expected_dir = [f"train_table_{i}.csv" for i in range(55)]

    assert len(what_indir) == 55
    assert {(frozenset(item)) for item in what_indir} == {(frozenset(item)) for item in expected_dir}
    
    my_teardown()


@pytest.mark.slow
def test_training_files_to_csv_each_csv_file_contents_as_expected():
    buildup_training_files_to_csv()

    files = os.listdir("./test_temp/")

    for file in files:
        curr_file = pd.read_csv("./test_temp/" + file)

        assert curr_file.shape == (3, 304)
        
        # Skipped testing for numerical values inside the files. 

    my_teardown()


def my_teardown():
    shutil.rmtree("./test_temp/")