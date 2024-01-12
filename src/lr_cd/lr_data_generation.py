# lr_data_generation.py
# author: Sam Fo
# date: 2024-01-10

def generate_data_lr(n, theta, random_seed=123):
    """Generate a number of data points base on the theta coefficients.

    Parameters
    ----------
    n : integer
        The number of data points.
    theta : float
        The true theta coefficient for the linear regression
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
    >>> generated_data = generate_data_lr(100, 3, 123)
    """
