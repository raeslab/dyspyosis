[![Run Pytest](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml) [![Coverage](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![PyPI version](https://badge.fury.io/py/dyspyosis.svg)](https://badge.fury.io/py/dyspyosis) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Dyspyosis

Python package that can be used to compute dysbiosis scores. The package leverages autoencoders based
anomaly detection. Further details on this method are available [here](https://github.com/raeslab/dyspyosis/blob/main/docs/method.md).

![A gumpy black snake, minimalist illustration](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/img/dyspyosis_logo_small.jpg)

## Installation

Before installing dyspyosis, ensure you have the CUDA toolkit v11.x and matching cuDNN installed, these are required for Tensorflow. Which version you need 
depends on your hardware, e.g. for a GTX 10XX you'll need [CUDA Toolkit 11.2] and the matching [cuDNN (8.1.1)], for
more recent cards you can get more recent versions.

Next, install dyspyosis using the command below.

```commandline
pip install dyspyosis
```

## Usage

Below you can find an example how to use the dyspyosis package. Note that this is for testing purposes and parameters 
have been set to complete the script quickly. For real data you'll want to increase the ```rarefication_count``` (the 
number of times samples will be rarefied) to a large number (the number of samples x rarefication_count should be > 10k) 
and increase the number of ```epochs``` to 4000.

The ```encode_dim``` is the size of the latent space and has been found to work best when set between 4 and 8 depending
on the number of genera in the input data, lower encoder_dim values working better with fewer genera. 

The loss, the main metric for dysbiosis, can be computed using ```compute_loss()```, while the laten space can be
accessed using ```get_latent```. See the example below.

**Note**: Depending on your system, you might need to set an environmental variable ```CUDA_VISIBLE_DEVICES``` to "0" before
loading dyspyosis to use the GPU. Try this in case CUDA is installed, but you get an error that no CUDA device was found.

**Note**: The neural network dyspyosis is based on is relatively small, depending on the complexity of your dataset and 
size of the latent space, running dyspyosis on CPU might outperform the GPU (see benchmarks)! To do so, set 
```CUDA_VISIBLE_DEVICES``` to "-1" and ```CUDA_DEVICE_ORDER``` to "PCI_BUS_ID" in your environment before launching 
dyspyosis.

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
        encode_dim=4
    )

    dyspyosis.run_training(epochs=5)

    loss = dyspyosis.compute_loss()
    loss.to_csv("./data/loss_out.tsv", sep=",", index=None)

    latent = dyspyosis.get_latent()
    latent.to_csv("./data/latent_out.tsv", sep=",", index=None)
```

## Benchmarks

There are two benchmark scripts included in the repository: ```benchmark_cpu.py``` and ```benchmark_gpu.py```. When
running the CPU benchmark it is important to set two environmental variables before running the code, ```CUDA_VISIBLE_DEVICES``` needs to be "-1"
and ```CUDA_DEVICE_ORDER``` needs to be "PCI_BUS_ID". This ensures that the CPU benchmark actually runs on the CPU in case a GPU is available.

Here are some results running dyspyosis on hardware we have access to.

| Type |                     Hardware | Epochs |       Time (s) |
|-----:|-----------------------------:|-------:|---------------:|
|  CPU |       Intel i5-7500 @ 3.4Ghz |    100 |       185.0017 |
|  CPU |            AMD Ryzen 7 3700X |    100 |       115.1882 |
|  GPU |  NVIDIA GeForce GTX 1060 6GB |    100 |       691.4091 |
|  GPU | NVIDIA GeForce RTX 4080 16GB |    100 |       340.6128 |

## For developers

To create the same environment the main devs are using, use [requirements.txt](https://github.com/raeslab/dyspyosis/blob/main/docs/dev/requirements.txt) to install
the exact versions off all packages.

Clone the repository, create a virtual environment and install all requirements first. Additionally, ensure you have
the CUDA toolkit v11.x and matching cuDNN installed, these are required for Tensorflow. Which version you need 
depends on your hardware, e.g. for a GTX 10XX you'll need [CUDA Toolkit 11.2] and the matching [cuDNN (8.1.1)], for
more recent cards you can get more recent versions.

```commandline
git clone https://github.com/raeslab/dyspyosis
cd dyspyosis
python -m venv venv
source venv/activate
pip install -r docs/dev/requirements.txt
```

To run tests, use the command below. There are a number of Deprecation Warnings (due to tensorflow) that can be
suppressed by ```--disable-warnings```.

```commandline
pytest tests/ --disable-warnings --cov=src --cov-report=term-missing --cov-report=xml
```

## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an [issue](https://github.com/raeslab/dyspyosis/issues).
  * Pull requests are welcome! Though open an [issue](https://github.com/raeslab/dyspyosis/issues) first to discuss which features/changes you wish to implement.

## Contact and License

dyspyosis was developed by [Sebastian Proost](https://sebastian.proost.science/) at the [RaesLab](https://raeslab.sites.vib.be/en) (part of [VIB](https://vib.be/en#/) and [KULeuven](https://www.kuleuven.be/english/kuleuven/index.html)). dyspyosis is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

For commercial access inquiries, please contact [Jeroen Raes](mailto:jeroen.raes@kuleuven.vib.be).

[CUDA Toolkit 11.2]: https://developer.nvidia.com/cuda-11.2.0-download-archive
[cuDNN (8.1.1)]: https://developer.nvidia.com/rdp/cudnn-archive
