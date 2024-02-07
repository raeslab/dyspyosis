[![Run Pytest](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml)[![Coverage](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Dyspyosis

Python package that can be used to compute dysbiosis scores. The package leverages autoencoders based
anomaly detection. Further details on this method are available [here](./docs/method.md).

![A gumpy black snake, minimalist illustration](./docs/img/dyspyosis_logo_small.jpg)

## Installation

Make sure you have [CUDA Toolkit 11.2] and the matching [cuDNN (8.1.1)] installed on your system (required for Tensorflow).

```commandline
pip install <TODO>
```

## Usage

Below you can find an example how to use the dyspyosis package. Note that this is for testing purposes and parameters 
have been set to complete the script quickly. For real data you'll want to increase the ```rarefication_count``` (the 
number of times samples will be rarefied) to a large number (the number of samples x rarefication_count should be > 10k) 
and increase the number of ```epochs``` to 4000.

```python
import pandas as pd
from dyspyosis import Dyspyosis

if __name__ == "__main__":
    df = pd.read_table("./data/test.tsv", index_col=0)

    dyspyosis = Dyspyosis(
        df.values,
        labels=df.index.tolist(),
        rarefication_depth=5000,
        rarefication_count=10,
    )

    dyspyosis.run_training(epochs=5)

    loss = dyspyosis.compute_loss()
    loss.to_csv("./data/loss_out.tsv", sep=",", index=None)
```

## Benchmarks

There are two benchmark scripts included in the repository: ```benchmark_cpu.py``` and ```benchmark_gpu.py```. When
running the CPU benchmark it is important to set two environmental variables before running the code, ```CUDA_VISIBLE_DEVICES``` needs to be "-1"
and ```CUDA_DEVICE_ORDER``` needs to be "PCI_BUS_ID". This ensures that the CPU benchmark actually runs on the CPU in case a GPU is available.

Here are some results running dyspyosis on hardware we have access too.

| Type |                    Hardware | Epochs | Time (s) |
|-----:|----------------------------:|-------:|---------:|
|  CPU |      Intel i5-7500 @ 3.4Ghz |    100 | 185.0017 |
|  GPU | NVIDIA GeForce GTX 1060 6GB |    100 | 691.4091 |

## For developers

Clone the repository, create a virtual environment and install all requirements first. Additionally, ensure you have
[CUDA Toolkit 11.2] and the matching [cuDNN (8.1.1)] installed on your system (required for Tensorflow).

```commandline
git clone https://github.com/raeslab/dyspyosis
cd dyspyosis
python -m venv venv
source venv/activate
pip install -r requirements.txt
```

To run tests, use the command below. There are a number of Deprecation Warnings (due to tensorflow) that can be
suppressed by ```--disable-warnings```.

```commandline
pytest tests/ --disable-warnings --cov=dyspyosis --cov-report=term-missing --cov-report=xml
```

[CUDA Toolkit 11.2]: https://developer.nvidia.com/cuda-11.2.0-download-archive
[cuDNN (8.1.1)]: https://developer.nvidia.com/rdp/cudnn-archive