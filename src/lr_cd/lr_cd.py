# lr_cd.py
# author: Andy Zhang
# date: 2024-01-18

import numpy as np

def coordinate_descent(X, y, ϵ=1e-6, max_iterations=1000):
    """
    Perform coordinate descent to minimize the mean square error of linear regression.

    The function takes the predictor `X` and response `y` with an initial guess for the intercept
    and coefficient weights vector. 
    The function can return the optimized intercept and coefficient weights vector, and 
    the number of iterations when the algorithm stops. 

    Parameters
    ----------
    X : ndarray
        Feature data matrix.
    y : ndarray
        Response data vector.
    ϵ : float, optional
        Stop the algorithm if the change in weights is smaller than the value (default is 1e-6).
    max_iterations : int, optional
        Maximum number of iterations (default is 1000).

    Returns
    -------
    intercept : float
        Optimized intercept.
    coef : ndarray
        Optimized coefficient weights vector.
    iteration : int
        The number of iterations when the algorithm stops.

    Raises
    ------
    TypeError
        If X or y is not a numpy array.
    ValueError
        If X and y do not have the same number of rows.

    Examples
    -------
    >>> import numpy as np
    >>> from lr_cd.lr_cd import coordinate_descent
    >>> X = np.array([-1.28162658, -0.99995041, -0.78869328, -0.47180759, -0.29575998, 
    ...               -0.22534094, 0.23238284,  0.30280188,  1.71118274, 1.81681131])
    >>> y = np.array([1.2390575, 1.99411649, 2.58284984, 2.37416463, 1.82673695,
    ...              1.71754177, 1.150911, 1.05020832, -0.28251291, -0.40102325])
    >>> intercept, coef, _ = coordinate_descent(X, y)
    >>> intercept
    0.42167642
    >>> coef
    array([[1.88190714]])
    """
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y should be a numpy array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y should have the same number of rows.")

    y=y.reshape(-1, 1)
    n = X.shape[0]
    coef = np.zeros(X.shape[1])
    intercept = 0.0
    n_features = X.shape[1]
    iterations = 1

    while iterations < max_iterations:
        coef_o = coef
        intercept_o = intercept

        intercept = np.mean(y - np.dot(X, coef))
        for j in range(n_features):
            tmp = y - np.dot(X, coef).reshape(-1, 1) - intercept * np.ones(n).reshape(-1, 1) + np.dot(X[:, j], coef[j]).reshape(-1, 1)
            
            numerator = X[:, j] @ tmp
            denominator = X[:, j] @ X[:, j]
            
            coef[j] = numerator[0] / denominator

        iterations += 1
        if np.linalg.norm(coef_o - coef) < ϵ and np.linalg.norm(intercept_o - intercept) < ϵ:
            break

    return intercept, coef.reshape(1, X.shape[1]), iterations
