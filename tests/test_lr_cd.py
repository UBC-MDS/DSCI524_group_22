

import pandas as pd
import pytest
import sys
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from lr_cd.lr_cd import coordinate_descent





def test_coordinate_descent_max_iterations():
    X = np.array([[-1.28162658, -0.99995041, -0.78869328, -0.47180759, -0.29575998,
                   -0.22534094, 0.23238284,  0.30280188,  1.71118274, 1.81681131]]).reshape(-1,1)
    y = np.array([1.2390575, 1.99411649, 2.58284984, 2.37416463, 1.82673695,
                  1.71754177, 1.150911, 1.05020832, -0.28251291, -0.40102325]).reshape(-1,1)
    max_iterations=2

    intercept, coef, iterations = coordinate_descent(X, y, max_iterations=2)

    assert iterations >= max_iterations




def test_coordinate_descent_j0():
    X = np.array([[-1.28162658, -0.99995041, -0.78869328, -0.47180759, -0.29575998,
                   -0.22534094, 0.23238284,  0.30280188,  1.71118274, 1.81681131]]).reshape(-1,1)
    y = np.array([1.2390575, 1.99411649, 2.58284984, 2.37416463, 1.82673695,
                  1.71754177, 1.150911, 1.05020832, -0.28251291, -0.40102325]).reshape(-1,1)

    intercept, coef, iterations = coordinate_descent(X, y)

    assert np.isclose(intercept, 1.325205, rtol=1e-6)

def test_coordinate_descent_j_greater_than_0():
    X = np.array([[-1.28162658, -0.99995041, -0.78869328, -0.47180759, -0.29575998,
                   -0.22534094, 0.23238284,  0.30280188,  1.71118274, 1.81681131]]).reshape(-1,1)
    y = np.array([1.2390575, 1.99411649, 2.58284984, 2.37416463, 1.82673695,
                  1.71754177, 1.150911, 1.05020832, -0.28251291, -0.40102325]).reshape(-1,1)

    intercept, coef, iterations = coordinate_descent(X, y)

    assert np.allclose(coef, np.array([[-0.829311]]), rtol=1e-6)





def test_lr_cd_input_X_type_error():
	np.random.seed(666666)
	X = 2 * np.random.rand(100, 5)
	y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
	y = y.reshape(-1, 1)
	with pytest.raises(TypeError):
	    coordinate_descent(1, y)


def test_lr_cd_input_y_type_error():
	np.random.seed(666666)
	X = 2 * np.random.rand(100, 5)
	y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
	y = y.reshape(-1, 1)
	with pytest.raises(TypeError):
	    coordinate_descent(X, 1)


def test_input_validation():

    with pytest.raises(TypeError):
        coordinate_descent(list(range(10)), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        coordinate_descent(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))


def test_coordinate_descent_true_if_statement():
	np.random.seed(666666)
	X = 2 * np.random.rand(100, 5)
	y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
	y = y.reshape(-1, 1)

	true = np.array([[ 3, 2, 1.5, 0.5, 0]])

	intercept, coef, iterations = coordinate_descent(X, y)

	assert iterations <= 1000

	assert pytest.approx(intercept, abs=1) == 4

	assert np.allclose(coef, true, atol=1)

	assert iterations > 0



def test_coordinate_descent_false_if_statement():

	np.random.seed(666666)
	X = 2 * np.random.rand(100, 5)
	y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
	y = y.reshape(-1, 1)
	true = np.array([[ 3, 2, 1.5, 0.5, 0]])


	intercept, coef, iterations = coordinate_descent(X, y)

	assert iterations <= 1000

	assert pytest.approx(intercept, abs=1) == 4

	assert np.allclose(coef, true, atol=1)

	assert iterations > 0









def test_lr_cd():
	np.random.seed(666666)
	X = 2 * np.random.rand(100, 5)
	y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
	y = y.reshape(-1, 1)

	lin_reg = LinearRegression()
	lin_reg.fit(X, y)
	intercept, coef, _ = coordinate_descent(X, y)

	assert isinstance(X, np.ndarray)
	assert isinstance(y, np.ndarray)
	assert isinstance(intercept, float)
	assert isinstance(coef, np.ndarray)

	expected_result_1 = lin_reg.intercept_
	calculated_result_1 = intercept
	expected_result_2 = lin_reg.coef_
	calculated_result_2 = coef

	assert np.allclose(expected_result_1, calculated_result_1, atol=1e-5)
	assert np.allclose(expected_result_2, calculated_result_2, atol=1e-5)


