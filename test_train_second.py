"""
Test files for train_second.py
Created on: 11 June 2021
"""

import pytest
from train_second import get_args

def test_parse_args_correctly():
    parsed = get_args(["--batch_size", "100", "--lr", "0.0001"])

    assert parsed.batch_size == 100
    assert parsed.lr == 0.0001