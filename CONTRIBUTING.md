# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Important prerequisites when contributing

The QuantFinLib code is published under the MIT license. You copyright notice can be
added to source file where you contributed substantionally.

### Report Bugs

Report bugs at https://github.com/quantfinlib/quantfinlib/issues.

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


### Write Documentation and Example Notebooks

QuantFinLib could always use more documentation and examples, whether as part of the
official QuantFinLib docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/quantfinlib/quantfinlib/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `quantfinlib` for local development.

1. Fork the `quantfinlib` repo on GitHub.
2. Clone your fork locally

    ```
    $ git clone git@github.com:quantfinlib/quantfinlib.git
    ```

3. Ensure [poetry](https://python-poetry.org/docs/) is installed.
4. Ensure that [pandocs](https://pandoc.org/installing.html) is installed.
5. Install dependencies and start your virtualenv:

    ```
    $ poetry install -E test -E doc -E dev
    ```
6. If you want to run jupter notebooks then install a kernel

    ```
    $ poetry run python -m ipykernel install --name quantfinlib-kernel
    ```

7. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

8. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```
    $ poetry run tox
    ```

9. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

10. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.9 and above. Check
4. https://github.com/quantfinlib/quantfinlib/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

```
$ poetry run pytest tests/test_some_item.py
```

To run a subset of tests.


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```
$ poetry run bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.


## Formatting the code

black is code formatter that helps you reformat the code such that it follows the PEP8 rules (see https://peps.python.org/pep-0008/). isort is a library that sorts your import statements in a consistent and logical order. 

We use precommit to automatically run black and isort on the quantfinlib library and reformat the source code if necessary. 
precommit, black, and isort libraries belong to the dev dependencies. So in order to have them installed in your poetry environment, make sure to run the following.

```
poetry install --with dev
```

Then runing `poetry run pre-commit install` will initialize a file named `pre-commit` in the following path `.git/hooks/`. Remove the content from that file and write the following content in that file:

```
#!/bin/sh
# Run isort on all files
poetry run isort .
# Run black on all files
poetry run python -m black quantfinlib

# Check if there are any changes
if ! git diff --quiet; then
  # Stage all changes
  git add .
  # Commit the changes with a message
  git commit -ammend -m "reformatted files with black"
fi
```

