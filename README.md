# lr_cd

A better implementation of the linear regression in Python! We are going to implement the linear regression by coordinate descent (CD) algorithm in convex optimization. Our package will have three major parts, including: 1. data generation, 2. coordinate descent algorithm, and 3. visualization.


There are three major functions in this package:
- data generation function
- coordinate descent algorithm function
- visualization function


Python package `scikit-learn` have the similar functionality. But the algorithm in our implementation is quite different from the existed one. See below links for more informaton of `sklearn.linear_model.LinearRegression`.

https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/linear_model/_base.py#L534


## Installation

```bash
$ pip install lr_cd
```

## Usage

We can use this package to find the coefficients of linear regression.

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
