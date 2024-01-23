import numpy as np
import matplotlib.pyplot as plt
import pytest
from lr_cd.lr_plotting import plot_lr

X_test = np.array([1, 2, 3])
X_test_2d = np.array([[1],[2],[3]])
y_test = np.array([1, 3, 4])
y_test_2d = np.array([[1],[3],[4]])
intercept_test = 0.0
coef_test = np.array([1])

# test if function return the correct data type
def test_plot_lr_returns_plot():
    result = plot_lr(X_test, y_test, intercept_test, coef_test)
    assert isinstance(result, plt.Figure), "`plot_lr` should return an figure object"

# Test that the function can handle 2D inputs for X
def test_plot_lr_input_X_2d():
    fig = plot_lr(X_test_2d, y_test, intercept_test, coef_test)
    assert isinstance(fig, plt.Figure), "`plot_lr` should return a matplotlib Figure object when given 2D input for X"

# Test that the function can handle 2D inputs for y
def test_plot_lr_input_y_2d():
    fig = plot_lr(X_test, y_test_2d, intercept_test, coef_test)
    assert isinstance(fig, plt.Figure), "`plot_lr` should return a matplotlib Figure object when given 2D input for y"

# Test handling of incorrect inputs    
def test_plot_lr_input_X_type_error():
    with pytest.raises(TypeError):
        plot_lr(1, y_test, intercept_test, coef_test)

def test_plot_lr_input_y_type_error():
    with pytest.raises(TypeError):
        plot_lr(X_test, 1, intercept_test, coef_test)

def test_plot_lr_input_intercept_type_error():
    with pytest.raises(TypeError):
        plot_lr(X_test, y_test, "", coef_test)

def test_plot_lr_input_coef_type_error():
    with pytest.raises(TypeError):
        plot_lr(X_test, y_test, intercept_test, "")

# Test handling of mismatched input sizes
def test_plot_lr_input_X_y_dimensions_value_error():
    with pytest.raises(ValueError):
        plot_lr(np.array([1, 2, 3]), np.array([1, 2, 3, 4]), intercept_test, coef_test)




