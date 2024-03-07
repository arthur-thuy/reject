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

```{toctree}
:maxdepth: 1
:caption: API Reference

autoapi/index
```

```{toctree}
:maxdepth: 1
:caption: User Guide

guide/classification_rejection.ipynb
guide/diversity_quality_score.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Other

changelog.md
contributing.md
```
