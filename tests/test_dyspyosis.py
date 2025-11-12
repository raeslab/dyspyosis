import pytest
import pandas as pd
import numpy as np
from dyspyosis import Dyspyosis


@pytest.fixture
def mock_data():
    """Generate mock data for testing."""
    return np.random.randint(0, high=1000, size=(100, 10))


@pytest.fixture
def mock_labels():
    """Generate mock labels for testing."""
    return [f"label_{i}" for i in range(100)]


@pytest.fixture
def dyspyosis_instance(mock_data, mock_labels):
    """Creates an instance of Dyspyosis with mock data."""
    return Dyspyosis(data=mock_data, labels=mock_labels, rarefication_depth=1000)


@pytest.fixture
def dyspyosis_labelless_instance(mock_data):
    """Creates an instance of Dyspyosis with mock data."""
    return Dyspyosis(data=mock_data, rarefication_depth=1000)


def test_initialization(dyspyosis_instance):
    """Test whether the Dyspyosis class initializes correctly with the given data."""
    assert dyspyosis_instance.data is not None, (
        "Data should not be None after initialization."
    )
    assert dyspyosis_instance.labels is not None, (
        "Labels should not be None after initialization."
    )
    assert dyspyosis_instance.autoencoder is not None, (
        "Autoencoder should be created upon initialization."
    )
    assert dyspyosis_instance.encoder is not None, (
        "Encoder should be created upon initialization."
    )
    assert dyspyosis_instance.decoder is not None, (
        "Decoder should be created upon initialization."
    )


def test_training(monkeypatch, dyspyosis_instance):
    """Test the training process of the autoencoder."""
    try:
        dyspyosis_instance.run_training(epochs=5, batch_size=32)
        training_passed = True
    except Exception:
        training_passed = False
    assert training_passed, "Training should run without errors."


def test_compute_loss(monkeypatch, dyspyosis_instance):
    """Test the loss computation of the autoencoder."""
    loss_output = dyspyosis_instance.compute_loss()

    assert isinstance(loss_output, pd.DataFrame), (
        "compute_loss should return a pandas DataFrame."
    )
    assert loss_output.shape == (100, 2), "DataFrame should be correct"
    assert "loss" in loss_output.columns, "DataFrame should contain a 'loss' column."
    assert "label" in loss_output.columns, (
        "DataFrame should contain a 'label' column if labels are provided."
    )

    # Remove labels to check output in the absense of labels
    dyspyosis_instance.labels = None
    loss_output = dyspyosis_instance.compute_loss()

    assert isinstance(loss_output, pd.DataFrame), (
        "compute_loss should return a pandas DataFrame."
    )
    assert loss_output.shape == (100, 1), "DataFrame should be correct"
    assert "loss" in loss_output.columns, "DataFrame should contain a 'loss' column."
    assert "label" not in loss_output.columns, (
        "DataFrame shouldn't contain a 'label' column if labels aren't provided."
    )


def test_get_latent(monkeypatch, dyspyosis_instance):
    latent = dyspyosis_instance.get_latent()

    assert isinstance(latent, pd.DataFrame), (
        "get_latent should return a pandas DataFrame."
    )
    assert latent.shape == (
        100,
        dyspyosis_instance.encode_dim + 1,
    ), "output should be the the number of samples by latent space + labels"
    assert "label" in latent.columns, (
        "DataFrame should contain a 'label' column if labels are provided."
    )

    # Remove labels to check output in the absense of labels
    dyspyosis_instance.labels = None
    latent = dyspyosis_instance.get_latent()
    assert isinstance(latent, pd.DataFrame), (
        "get_latent should return a pandas DataFrame."
    )
    assert latent.shape == (
        100,
        dyspyosis_instance.encode_dim,
    ), "output should be the the number of samples by latent space"
    assert "label" not in latent.columns, (
        "DataFrame should not contain a 'label' column if no labels are provided."
    )
