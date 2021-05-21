"""
Unit Test for txt_to_csv_noisy.py
"""
import pytest
from txt_to_csv_noisy import *

noisy_path = "../Example_data/noisy_train"
params_path = "../Example_data/params_train"
noisy_test_path = "../Example_data/noisy_test"

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
    pass


def test_parse_arguments_default_work_correctly():
    parsed = parse_arguments([])

    assert parsed.num_process == os.cpu_count()
    assert parsed.save_folder == "./csv_files/"
    assert parsed.noisy_path == "./data/training_set/noisy_train/"
    assert parsed.noisy_test_path == "./data/test_set/noisy_test/"
    assert parsed.params_path == "./data/training_set/params_train/"


@pytest.mark.skip
def test_think_of_name_later_presence_of_folder():
    pass


@pytest.mark.skip
def test_think_of_name_later_presence_of_all_csv_files():
    pass


@pytest.mark.skip
def test_think_of_name_later_each_csv_file_contents_as_expected():
    pass