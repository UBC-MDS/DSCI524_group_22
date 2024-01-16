

import pandas as pd
import pytest
import sys
import os

#import unittest
import numpy as np
from sklearn.linear_model import LinearRegression

from lr_cd.lr_cd import coordinate_descent


np.random.seed(666666)
X = 2 * np.random.rand(100, 5)
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(100)
y = y.reshape(-1, 1)



# Test for correct return type
def test_lr_cd():

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









