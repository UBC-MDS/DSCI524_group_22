# lr_data_generation.py
# author: Sam Fo
# date: 2024-01-10

import numpy as np


def generate_data_lr(n, n_features, theta, noise=0.2, random_seed=123):
    """Generate a number of data points base on the theta coefficients.

    Parameters
    ----------
    n : integer
        The number of data points.
    n_features : integer
        The number of features to generate, excluding the intercept.
    theta : ndarray
        The true scalar intercept and coefficient weights vector.
        The first element should always be the intercept.
    noise : float
        The standard deviation of a normal distribution added to the generated target y array as noise.
    random_seed : integer
        Random seed to ensure reproducibility.

    Returns
    -------
    X : ndarray
        Feature data matrix of shape (n_samples, n_features).
    y : ndarray
        Response data matrix of shape (n_samples, 1).

    Examples
    --------
    >>> from lr_cd.lr_data_generation import generate_data_lr
    >>> theta = np.array([4, 3])
    >>> generate_data_lr(n=10, n_features=1, theta=theta)
    """
    np.random.seed(random_seed)

    if not isinstance(n, int):
        raise ValueError('Sample size n must be an integer')

    if not isinstance(n_features, int):
        raise ValueError('Number of features must be an integer')

    if len(theta) != n_features + 1:
        raise ValueError('Number of features does not match with theta')

    if len(theta) < 2:
        raise ValueError('Insufficient number of elements in theta')

    X = np.random.random(size=n * n_features).reshape(n_features, n)
    true_intercept = theta[0]
    true_coeff = theta[1:].reshape(n_features, -1)
    noise = np.random.normal(
        loc=0.0, scale=noise, size=n)
    y = np.sum(X * true_coeff, axis=0) + true_intercept + noise
    return X.T, y.reshape(1, -1).T
