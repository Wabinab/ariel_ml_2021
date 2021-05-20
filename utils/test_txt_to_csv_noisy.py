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
    parsed = parse_arguments(["--num_process", "2", "--save_folder", "./csv_files"])

    assert parsed.num_process == 2
    assert parsed.save_folder == "./csv_files"

def test_parse_arguments_default_work_correctly():
    parsed = parse_arguments([])

    assert parsed.num_process == os.cpu_count()
    assert parsed.save_folder == "./csv_files/"