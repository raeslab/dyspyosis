from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import losses


def create_autoencoder(input_shape, encoding_dim=4, regularization_value=10e-5):
    """
    Creates an autoencoder and its corresponding encoder and decoder models.

    Parameters:
    -----------
    input_shape : int
        The number of features in the input data.
    encoding_dim : int, optional
        The size of the encoding layer. Default is 4.
    regularization_value : float, optional
        The L1 regularization factor. Default is 1e-5.

    Returns:
    --------
    autoencoder : keras.models.Model
        The assembled autoencoder model, compiled and ready for training.
    encoder : keras.models.Model
        The encoder part of the autoencoder.
    decoder : keras.models.Model
        The decoder part of the autoencoder.
    """
    # Setup Layers
    input_data = Input(shape=(input_shape,))
    encoded = Dense(
        encoding_dim,
        activation="relu",
        activity_regularizer=regularizers.l1(regularization_value),
    )(input_data)

    decoded = Dense(input_shape, activation="softmax")(encoded)

    # Create Autoencoder
    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")

    # Create Encoder
    encoder = Model(input_data, encoded)

    # Create Decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return autoencoder, encoder, decoder


def get_latent(encoder, data):
    latent = encoder.predict(data)

    return latent


def get_loss(autoencoder, data):
    predicted = autoencoder.predict(data)
    loss_function = losses.MeanSquaredError(reduction="none")

    output = [loss_function(a, b).numpy() for a, b in zip(predicted, data)]

    return output
