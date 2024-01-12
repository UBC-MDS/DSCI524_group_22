# lr_cd

A better implementation of the linear regression in Python! We are going to implement the linear regression by coordinate descent (CD) algorithm in convex optimization. Our package will have three major parts, including 1. data generation, 2. coordinate descent algorithm, and 3. visualization.


## Functions

There are three major functions in this package:
- `generate_data_lr(n, theta, random_seed=123)`: this function generates many random data points based on the theta coefficients, which will later be used for model fitting.
- `coordinate_descent(X, y, alpha, ϵ=1e-4, max_iterations=1000)`: this function performs coordinate descent to minimize the mean square error of linear regression and therefore outputs the optimized intercept and coefficient weights vector.
- `plot_lr(X, y, intercept, coef, plot_to)`: this function returns a scatter plot of the observed data points overlayed with a regression with optimized weights.


## Existed Package
Python package `scikit-learn` has a similar functionality. However, we are taking a different algorithm in our implementation and we believe it will be a better implementation. LinearRegression of Scikit-learn's contains a few optimization functions: `scipy.linalg.lstsq`, `scipy.sparse.linalg.lsqr` and `scipy.optimize.nnls` which rely on the singular value decomposition of feature matrix X. 

See the below links for more information on `sklearn.linear_model.LinearRegression`.

https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/linear_model/_base.py#L534


## Installation

```bash
$ pip install lr_cd
```

## Usage

We can use this package to find the coefficients of linear regression.

Example usage:
```
>>> from lr_cd.lr_cd import coordinate_descent
>>> model = coordinate_descent(X, y, alpha=0.01)
```

```
model.intercept_
0.42167642

model.coef_
array([1.88190714])
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`lr_cd` was created by Sam Fo, Jing Wen, Andy Zhang. It is licensed under the terms of the MIT license.

## Contributors

- Sam Fo for data generation
- Jing Wen for visualization
- Andy Zhang for algorithm



## Credits

`lr_cd` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
