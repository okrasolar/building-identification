# Building Identification

This is a work in progress project to use [Sentinel](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2/)
data to identify buildings.

See the [scripts](scripts) for more information on how to run different parts of the pipeline.

## Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `okra-buildings` with all the necessary packages to run the code. To 
activate this environment, run

```bash
conda activate okra-buildings
```

Running this code also requires you to sign up to [Earth Engine](https://developers.google.com/earth-engine/). Once you 
have done so, active the `okra-buildings` environment and run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine).

## Dev

[Black](https://black.readthedocs.io/en/stable/) is used for code formatting. 

[mypy](http://mypy-lang.org/) is used for type hints.

[pytest](https://docs.pytest.org/en/latest/) is our test runner.

These can be run with the following commands (from the project root):

```bash
black .
mypy src
pytest
```
