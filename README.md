# reject

<p align="center">
<a href="https://pypi.org/project/reject/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/reject">
    </a>
<img src="https://github.com/arthur-thuy/reject/actions/workflows/ci.yml/badge.svg" />
<a href='https://reject.readthedocs.io/en/latest/'>
        <img src='https://img.shields.io/readthedocs/reject' alt='Documentation Status' />
    </a>
<a href="https://app.codecov.io/gh/arthur-thuy/reject" >
 <img src="https://codecov.io/gh/arthur-thuy/reject/graph/badge.svg?token=wYnaStSR3z"/>
 </a>
<a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

`reject` is a Python library for _classification with rejection_. Neural networks are often confidently wrong when confronted with out-of-distribution data. When the prediction's uncertainty is too high, the model abstains from predicting and the observation is passed on to a human expert who takes the final decision. It is useful for applications where making an error can be more costly than asking a human expert for help.

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

User guide notebooks are provided in the [reject.readthedocs.io](http://reject.readthedocs.io/) documentation.
