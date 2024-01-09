[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Dyspyosis

Python package that can be used to compute dysbiosis scores.

![A gumpy black snake, minimalist illustration](./docs/img/dyspyosis_logo_small.jpg)

## Installation

Make sure you have CUDA Toolkit 11.2 and the matching cuDNN (8.1.1) installed on your system (required for Tensorflow).

```commandline
pip install <TODO>
```
## Usage



## For developers

Clone the repository, create a virtual environment and install all requirements first. Additionally, ensure you have
CUDA Toolkit 11.2 and the matching cuDNN (8.1.1) installed on your system (required for Tensorflow).

```commandline
git clone <URL>
cd <dir>
python -m venv venv
source venv/activate
pip install -r requirements.txt
```

To run tests, use the command below. There are a number of Deprecation Warnings (due to tensorflow) that can be
suppressed by ```--disable-warnings```.

```commandline
pytest tests/ --disable-warnings --cov=dyspyosis --cov-report=term-missing --cov-report=xml
```
