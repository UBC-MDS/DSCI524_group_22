# lr_plotting.py
# author: Jing Wen
# date: 2024-01-09

def plot_lr(data, est_val):
    """Visualize the "lr_cd" linear regression model.

    This function takes actual data points and an estimated regression line,
    displaying them together in a scatter plot. The plot is also
    saved as a PNG image.
    
    Parameters
    ----------
    data : DataFrame
        The observed data, with 'x' being the independent variable and 'y' being the dependent variable.
        Both 'x' and 'y' should be continuous and of the same length. The 'x' values should match
        those in `est_val`.
    est_val : dataframe
        The estimated values produced by the "lr_cd" linear regression model. This DataFrame
        should contain two columns: 'x', which matches the 'x' from `data`, and 'y_est', which
        contains the predicted 'y' values corresponding to each 'x'.

    Returns
    -------
    A PNG image named 'linear_regression_plot.png' is saved in the current directory.

    Examples
    --------
    >>> from lr_cd.lr_plotting import plot_lr
    >>> data = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 5]})
    >>> est_val = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [0.8, 2.1, 3, 4]})
    >>> plot_lr(data, est_val)
    """
    