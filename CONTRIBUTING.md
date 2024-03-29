# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `reject` for local development.

1. Clone the `reject` repository locally.
2. Install `reject` using `poetry`:

    ```console
    $ poetry install
    ```

3. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

4. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

5. Commit your changes and open a pull request.

## Pre-commit hooks

A handful of [pre-commit](https://pre-commit.com) hooks are provided, and ensuring their checks pass pass is highly recommended as they are later ran as part of CI checks that will block PRs.

These hooks include:

* Trailing whitespace trimming.
* Ensure EOF newline.
* Code formatting ([black](https://github.com/psf/black)).
* PEP8 linting ([flake8](https://github.com/pycqa/flake8)).
* Import sorting ([isort](https://github.com/PyCQA/isort)).

All *pre-commit* hooks can be installed with:

```shell
pre-commit install
```

Alternatively, all of the aforementioned tools are installed with the *dev* dependency group and can ran individually, without *pre-commit* hooks. For example, to format files with *black*:

```shell
poetry run black reject/
```

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.
