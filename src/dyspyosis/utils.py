import numpy as np


def rarefy(data, rarefication_depth=None, seed=0):
    """
    This function performs rarefaction on a matrix data.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the data.
    rarefication_depth : int, optional
        Sampling rarefication_depth. If not specified, the minimum number of occurrences is used.
    seed : int, optional
        Seed for the random number generator. Default is 0.

    Returns:
    --------
    output : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the rarefied data.
    """
    prng = np.random.default_rng(seed)
    noccur = np.sum(data, axis=1)
    nvar = data.shape[1]

    if rarefication_depth is None:
        rarefication_depth = np.min(noccur)
    elif rarefication_depth > np.min(noccur):
        print(
            f"Warning: Specified rarefication_depth ({rarefication_depth}) is larger than the minimum number of occurrences ({np.min(noccur)})."
        )

    output = np.empty((data.shape[0], nvar))
    for i in np.arange(data.shape[0]):  # for each sample
        p = data[i] / float(noccur[i])  # relative frequency / probability
        choice = prng.choice(nvar, rarefication_depth, p=p)
        output[i] = np.bincount(choice, minlength=nvar)

    return output


def scale_data(data, rarefication_depth):
    """
    Scales each row of the input data by the rarefaction depth.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the data to be scaled.
    rarefication_depth : float
        The value by which each value in the data matrix will be divided.

    Returns:
    --------
    scaled_data : numpy.ndarray
        The resulting data after scaling, which has the same shape as the input data.
    """

    # Ensure that data is a NumPy array to apply operations element-wise
    data = np.asarray(data)

    # Scale each row of the data matrix by the rarefaction depth
    scaled_data = data / rarefication_depth

    return scaled_data


def build_dataset(data, rarefication_depth, iterations=10, seed=0):
    """
    This function builds an expanded dataset by performing multiple rarefaction on a matrix data.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the data.
    rarefication_depth : int
        Number of reads to sample during rarefication.
    iterations : int, optional
        Number of rarefied sets to generate. Default is 10.
    seed : int, optional
        Seed for the random number generator. Default is 0.

    Returns:
    --------
    output : numpy.ndarray
        A 2D array of shape (n_samples * iterations, n_features) containing the rarefied data.
    """
    rarefied_sets = [
        rarefy(data, rarefication_depth, seed=seed + i) for i in range(iterations)
    ]
    return np.concatenate(rarefied_sets, axis=0)
