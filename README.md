[![Run Pytest](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/dyspyosis/actions/workflows/autopytest.yml) [![Coverage](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/dyspyosis/main/docs/coverage-badge.svg) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Dyspyosis

Python package that can be used to compute dysbiosis scores.

![A gumpy black snake, minimalist illustration](./docs/img/dyspyosis_logo_small.jpg)

## Installation

Make sure you have [CUDA Toolkit 11.2] and the matching [cuDNN (8.1.1)] installed on your system (required for Tensorflow).

```commandline
pip install dyspyosis
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

## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an [issue](https://github.com/raeslab/dyspyosis/issues).
  * Pull requests are welcome! Though open an [issue](https://github.com/raeslab/dyspyosis/issues) first to discuss which features/changes you wish to implement.

## Contact and License

dyspyosis was developed by [Sebastian Proost](https://sebastian.proost.science/) at the [RaesLab](https://raeslab.sites.vib.be/en) (part of [VIB](https://vib.be/en#/) and [KULeuven](https://www.kuleuven.be/english/kuleuven/index.html)). dyspyosis is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International]((https://creativecommons.org/licenses/by-nc-sa/4.0/)) license.

