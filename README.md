# reject

Functionalities for classification with rejection.

## Installation

```bash
$ pip install reject
```

## Documentation

The documentation is deployed to [reject.readthedocs.io](http://reject.readthedocs.io/).

## Usage

```python
from reject.reject import ClassificationRejector

y_pred  # Array of predictions. Shape (n_observations, n_classes) or (n_observations, n_samples, n_classes).
y_true  # Array of true labels. Shape (n_observations,).

# initialize the rejector
rej = ClassificationRejector(y_true_all, y_pred_all)
```
```python
# single rejection point
rej.reject(threshold=0.5, unc_type="TU", relative=True, show=True)
```
```bash
             Non-rejected    Rejected
---------  --------------  ----------
Correct               891          20
Incorrect             109         980

  Non-rejected accuracy    Classification quality    Rejection quality
-----------------------  ------------------------  -------------------
                 0.8910                    0.9355              40.9908
```

```python
# rejection curve
fig = rej.plot_reject(unc_type="TU", metric="NRA")
print(fig)
```

<img src="https://github.com/arthur-thuy/reject/assets/57416568/6a59f37a-0f2f-4a2c-96d8-8690b8e19df7" height="200"/>

An example notebook is provided, which can be found in the "Example usage" section of the documentation.


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`reject` was created by Arthur Thuy. It is licensed under the terms of the Apache License 2.0 license.

## Credits

`reject` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
