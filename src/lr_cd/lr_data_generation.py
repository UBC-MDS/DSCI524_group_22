# lr_data_generation.py
# author: Sam Fo
# date: 2024-01-10

import numpy as np


def generate_data_lr(n, n_features, theta, random_seed=123):
    """Generate a number of data points base on the theta coefficients.

    Parameters
    ----------
    n : integer
        The number of data points.
    n_features : ndarray
        The number of features to generate, excluding the intercept.
    theta : ndarray
        The true scalar intercept and coefficient weights vector.
        The first element should always be the intercept.
    random_seed : integer
        Random seed to ensure reproducibility.

    Returns
    -------
    X : ndarray
        Feature data matrix of shape (n_samples, n_features).
    y : ndarray
        Response data vector of shape (n_samples,).

    Examples
    --------
    >>> from lr_cd.lr_data_generation import generate_data_lr
    >>> theta = np.array([4, 3])
    >>> generate_data_lr(n=10, n_features=1, theta=theta)
    """
    np.random.seed(random_seed)

    if len(theta) != n_features + 1:
        raise ValueError('Number of features does not match with theta.')

    X = np.random.normal(size=n * n_features).reshape(n, n_features)
    true_intercept = theta[0]
    true_coeff = theta[1:]
    y = X * true_coeff.reshape(n_features, -1) + true_intercept
    return X, y
