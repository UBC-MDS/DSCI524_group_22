# lr_cd

A better implementation of the linear regression in Python! We are going to implement the linear regression by coordinate descent (CD) algorithm. Our package will have three major parts, including 1. data generation, 2. coordinate descent algorithm, and 3. visualization. Please refer to the link for additional details about the [coordinate descent (CD) algorithm](https://en.wikipedia.org/wiki/Coordinate_descent) if you are unfamiliar with it.



## Functions

There are three major functions in this package:

- `generate_data_lr(n, theta, random_seed=123)`: this function generates many random data points based on the theta coefficients, which will later be used for model fitting.
- `coordinate_descent(X, y, ϵ=1e-6, max_iterations=1000)`: this function performs coordinate descent to minimize the mean squared error of linear regression and therefore outputs the optimized intercept and coefficients vector.
- `plot_lr(X, y, intercept, coef)`: this function returns a scatter plot of the observed data points overlayed with a regression with optimized intercept and coefficients vector.


## Python Ecosystem Context
`LinearRegression` in Python package `scikit-learn` has a similar functionality. However, we use a different algorithm in the implementation and believe it will be a better one. `sklearn.linear_model.LinearRegression` contains a few optimization functions: `scipy.linalg.lstsq`, `scipy.sparse.linalg.lsqr` and `scipy.optimize.nnls` which basically rely on the singular value decomposition of feature matrix X. 

See the links for more information on [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).


## Installation

```bash
poetry install
```

## Usage

We can use this package to find the optimized intercept and coefficients vector of linear regression.

Example usage:

```
>>> from lr_cd.lr_data_generation import generate_data_lr
>>> theta = np.array([4, 3])
>>> X, y = generate_data_lr(n=10, n_features=1, theta=theta)

>>> from lr_cd.lr_cd import coordinate_descent
>>> intercept, coef, _ = coordinate_descent(X, y)

>>> from lr_cd.lr_plotting import plot_lr
>>> plot_lr(X, y, intercept, coef)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`lr_cd` was created by Sam Fo, Jing Wen, Andy Zhang. It is licensed under the terms of the MIT license.

## Contributors

- Sam Fo for data generation
- Andy Zhang for algorithm
- Jing Wen for visualization

## Credits

`lr_cd` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
