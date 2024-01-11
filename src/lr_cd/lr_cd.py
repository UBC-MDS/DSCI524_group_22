# lr_cd.py
# author: Andy Zhang
# date: 2024-01-10


def coordinate_descent(X, y, w, alpha, ϵ=1e-4, max_iterations=1000):
    """
    Perform coordinate descent to minimize the mean square error of linear regression.

    The function takes the predictor `X` and reponse `y` with initial guess for weights vector.
    With an appropriate learning rate `alpha`, the function can return the optimized weights vector. 


    Parameters
    ----------
    X : ndarray
        Feature data matrix.
    y : ndarray
        Response data vector.
    w : ndarray
        Initial weights vector.
    alpha : float
        Learning rate.
    ϵ : float, optional
        Stop the algorithm if the change in weights is smaller than the value (default is 1e-4).
    max_iterations : int, optional
        Maximum number of iterations (default is 1000).

    Returns
    -------
    ndarray
        Optimized weights vector.



    Examples
    -------
    >>> from lr_cd.lr_cd import coordinate_descent
    >>> X = array([[ 1.        , -1.28162658],
    ...            [ 1.        , -0.99995041],
    ...            [ 1.        , -0.78869328],
    ...            [ 1.        , -0.47180759],
    ...            [ 1.        , -0.29575998],
    ...            [ 1.        , -0.22534094],
    ...            [ 1.        ,  0.23238284],
    ...            [ 1.        ,  0.30280188],
    ...            [ 1.        ,  1.71118274],
    ...            [ 1.        ,  1.81681131]])
    >>> y = array([ 1.2390575 ,  1.99411649,  2.58284984,  2.37416463,  1.82673695,
    ...             1.71754177,  1.150911  ,  1.05020832, -0.28251291, -0.40102325])
    >>> coordinate_descent(X, y, w=np.array([0.0, 0.0]), alpha=0.01)
    array([ 1.88190714,  0.42167642])
    """

    pass  
