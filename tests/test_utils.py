import numpy as np
from dyspyosis.utils import rarefy, scale_data, build_dataset


def test_rarefy(capsys):
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Test with default rarefaction depth
    output = rarefy(data)
    assert np.all(np.sum(output, axis=1) == np.min(np.sum(data, axis=1)))

    # Test with specified rarefaction depth
    output = rarefy(data, rarefication_depth=2)
    assert np.all(np.sum(output, axis=1) == 2)

    # Test with rarefaction depth larger than the minimum sum of occurrences in data
    output = rarefy(data, rarefication_depth=10)
    assert np.all(np.sum(output, axis=1) == 10)

    # Check if a warning message is output
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_scale_data():
    data = np.array([[2, 4, 6], [10, 20, 30]])
    rarefication_depth = 2
    expected_scaled_data = np.array([[1, 2, 3], [5, 10, 15]])

    # Perform scaling on the data
    scaled_data = scale_data(data, rarefication_depth)

    # Validate the scaled data against the expected results
    assert np.array_equal(scaled_data, expected_scaled_data), (
        "Incorrect scaling of data"
    )


def test_build_dataset():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    iterations = 10
    rarefication_depth = 2

    # Test dataset building with specified rarefaction depth and iterations
    output = build_dataset(data, rarefication_depth, iterations=iterations, seed=0)

    # Validate output dimensions and sum per row
    assert output.shape == (iterations * data.shape[0], data.shape[1])
    assert np.all(np.sum(output, axis=1) == rarefication_depth)
