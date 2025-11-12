import numpy as np
from unittest.mock import MagicMock
from tensorflow.keras import backend as K
from tensorflow.keras import models

# Assuming create_autoencoder was defined in a module called autoencoder_module
from dyspyosis.autoencoder import create_autoencoder, get_loss


def test_autoencoder_creation():
    input_shape = 10
    encoding_dim = 4
    regularization_value = 1e-5

    autoencoder, encoder, decoder = create_autoencoder(
        input_shape, encoding_dim, regularization_value
    )

    # Test if the autoencoder, encoder and decoder models are created correctly
    assert autoencoder is not None, "Autoencoder model should not be None"
    assert encoder is not None, "Encoder model should not be None"
    assert decoder is not None, "Decoder model should not be None"

    # Check model input and output shapes
    assert autoencoder.input_shape == (
        None,
        input_shape,
    ), "Autoencoder input shape is incorrect"
    assert autoencoder.output_shape == (
        None,
        input_shape,
    ), "Autoencoder output shape is incorrect"
    assert encoder.output_shape == (
        None,
        encoding_dim,
    ), "Encoder output shape is incorrect"
    assert decoder.input_shape == (
        None,
        encoding_dim,
    ), "Decoder input shape is incorrect"

    # Check if the models have been compiled
    assert autoencoder.optimizer is not None, "Autoencoder has not been compiled"

    # Clear the session to avoid clutter from old models / layers.
    K.clear_session()


def test_get_loss():
    # Mock data and prediction
    data = np.array([[0, 0], [1, 1]])
    predicted_data = np.array([[0.5, 0.5], [0.5, 0.5]])

    # Mock the autoencoder's predict method to return the predicted_data
    mock_autoencoder = MagicMock(spec=models.Model)
    mock_autoencoder.predict.return_value = predicted_data

    # Use the mocked autoencoder in the get_loss function
    losses_output = get_loss(mock_autoencoder, data)

    # Expected MSE values:
    # Sample 0: MSE([0.5, 0.5], [0, 0]) = mean((0.5-0)^2 + (0.5-0)^2) = mean(0.25 + 0.25) = 0.25
    # Sample 1: MSE([0.5, 0.5], [1, 1]) = mean((0.5-1)^2 + (0.5-1)^2) = mean(0.25 + 0.25) = 0.25
    expected = [0.25, 0.25]
    assert np.allclose(losses_output, expected), (
        f"Expected {expected}, got {losses_output}"
    )
