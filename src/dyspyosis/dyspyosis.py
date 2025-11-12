from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional

from .utils import build_dataset, rarefy, scale_data
from .autoencoder import create_autoencoder, get_loss, get_latent


class Dyspyosis:
    """
    A class for creating and training an autoencoder model to analyze dysbiosis data.

    Attributes:
    -----------
    data : pd.DataFrame
        The dataset used for training and evaluating the autoencoder.
    labels : list, optional
        The labels corresponding to the dataset, added when calculating losses per label.
    rarefication_depth : int
        Depth to rarefy to when generating training data.
    rarefication_count : int
        The number of times the data is rarefied when generation the training data
    seed : int
        The random state seed used for data splitting and rarefication.

    Methods:
    --------
    run_training(epochs, batch_size)
        Trains the autoencoder using the scaled and rarefied data.
    compute_loss()
        Computes the reconstruction loss of the autoencoder model on the scaled data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: Optional[list] = None,
        rarefication_depth: int = 5000,
        rarefication_count: int = 10,
        encode_dim: int = 4,
        seed: int = 0,
    ):
        """
        Initializes the Dyspyosis class with data, optional labels, and rarefication parameters.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to be used in the analysis.
        labels : list, optional
            Optional labels corresponding to the dataset.
        rarefication_depth : int
            Number of reads to rarefy to.
        rarefication_count : int
            The number of times to rarefy the data to generate training data
        encode_dim: int
            Number of dimensions the latent space should have
        seed : int
            The random state seed for reproducibility purposes.
        """
        self.data = data
        self.labels = labels
        self.rarefication_depth = rarefication_depth
        self.rarefication_count = rarefication_count
        self.encode_dim = encode_dim
        self.seed = seed

        self.x_test = None
        self.x_train = None

        self.scaled_data = scale_data(
            rarefy(data, rarefication_depth, seed=seed), self.rarefication_depth
        )
        self.autoencoder, self.encoder, self.decoder = create_autoencoder(
            self.data.shape[1], encoding_dim=self.encode_dim
        )

        full_data = scale_data(
            build_dataset(
                self.data,
                self.rarefication_depth,
                self.rarefication_count,
                seed=self.seed + 1,
            ),
            self.rarefication_depth,
        )

        self.x_train, self.x_test = train_test_split(
            full_data, test_size=0.15, random_state=self.seed
        )

    def run_training(self, epochs: int = 4000, batch_size: int = 64) -> None:
        """
        Trains the autoencoder using the prepared training data.

        Parameters:
        -----------
        epochs : int
            The number of epochs to train the autoencoder.
        batch_size : int
            The batch size used during training.
        """
        self.autoencoder.fit(
            self.x_train,
            self.x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.x_test, self.x_test),
        )

    def compute_loss(self) -> pd.DataFrame:
        """
        Computes the reconstruction loss of the autoencoder on the scaled data.

        Returns:
        --------
        output : pd.DataFrame
            A dataframe with loss values and optional labels.
        """
        loss = get_loss(self.autoencoder, self.scaled_data)

        if self.labels is not None:
            output = pd.DataFrame({"label": self.labels, "loss": loss})
        else:
            output = pd.DataFrame({"loss": loss})

        return output

    def get_latent(self) -> pd.DataFrame:
        """
        Retrieves the latent representations (encoded features) of the scaled data using the encoder part of the autoencoder model.

        Returns:
        --------
        latent : pd.DataFrame
            A DataFrame containing the latent representations of the scaled data.
        """
        latent = get_latent(self.encoder, self.scaled_data)

        num_columns = latent.shape[1]
        output = pd.DataFrame(latent, columns=[f"L{i + 1}" for i in range(num_columns)])

        if self.labels is not None and len(self.labels) == output.shape[0]:
            output["label"] = self.labels

        return output
