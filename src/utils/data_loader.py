"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist


def load_dataset(dataset_name="mnist", flatten=True, normalize=True):
    """
    Load and preprocess the specified dataset between mnist and fashion-mnist. Has options to flatten images and normalize pixel values.
    """

    if dataset_name.lower() == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name.lower() in ["fashion_mnist", "fashion-mnist"]:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")

    # Convert to float32 to avoid overflow issues with uint8
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Normalize for better training stability
    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    # Flatten for fully connected network
    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test