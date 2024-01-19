# test_data_generation.py

import numpy as np
import pytest
from lr_cd.lr_data_generation import *


theta = np.array([4, 3])
n = 5
n_features = 1


def test_non_integer_n():
    with pytest.raises(ValueError, match="Sample size n must be an integer"):
        generate_data_lr(0.5, n_features, theta)


def test_non_integer_n_features():
    with pytest.raises(ValueError, match="Number of features must be an integer"):
        generate_data_lr(n, 0.5, theta)


def test_length_theta_matches_features():
    with pytest.raises(ValueError, match="Number of features does not match with theta"):
        generate_data_lr(n, n_features, np.array([4, 3, 2, 1]))


def test_theta_at_least_2_elements():
    with pytest.raises(ValueError, match="Insufficient number of elements in theta"):
        generate_data_lr(n, 0, np.array([4]))


def test_correct_result_X():
    test_X = generate_data_lr(n, n_features, theta)[0]
    expected_X = np.array([[0.69646919],
                           [0.28613933],
                           [0.22685145],
                           [0.55131477],
                           [0.71946897]])
    assert np.allclose(test_X, expected_X)


def test_correct_result_y():
    test_y = generate_data_lr(n, n_features, theta)[1]
    expected_y = np.array([[6.15382877],
                           [4.84811446],
                           [4.63971417],
                           [6.04981399],
                           [5.8345469]])
    assert np.allclose(test_y, expected_y)


def test_correct_dimenstion_result():
    result = generate_data_lr(n, n_features, theta)
    assert len(result) == 2


def test_result_is_array():
    test_X, test_y = generate_data_lr(n, n_features, theta)
    assert isinstance(test_X, np.ndarray)
    assert isinstance(test_y, np.ndarray)
