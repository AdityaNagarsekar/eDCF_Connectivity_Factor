from typing import Optional
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

class IrisVersicolorData:
    """
    Encapsulates data from the Iris Versicolor class with functionalities to perform PCA for dimensionality reduction.

    This class allows extraction of Iris Versicolor data after applying PCA to the whole Iris dataset.
    """

    def __init__(self, identity: int = 2, pca_components: int = 2):
        """
        Constructor for IrisVersicolorData.

        :param identity: The class identity (label) for Iris Versicolor. Default is 1 as per sklearn's labeling.
        :param pca_components: Number of principal components for PCA. Default is 2.
        """
        self.__identity = identity  # Class identity (label) for Iris Versicolor
        self.__features = []             # Feature names
        self.__datapoints_original = np.ndarray([])  # Original 4D data points for all Iris classes
        self.__datapoints_pca = np.ndarray([])       # PCA-transformed data points
        self.__labels = np.ndarray([])   # Target labels for all Iris classes
        self.__pca_components = pca_components       # Number of PCA components

        self.__load_data()
        self.__perform_pca()

    def __load_data(self):
        """
        Private method to load the entire Iris dataset.
        """
        iris = load_iris()
        self.__features = iris.feature_names  # Extract feature names
        self.__datapoints_original = iris.data  # Load all the data (4D)
        self.__labels = iris.target            # Load the labels (0: Setosa, 1: Versicolor, 2: Virginica)

    def __perform_pca(self):
        """
        Private method to perform PCA on the original data points.
        """
        if self.__pca_components <= 0 or self.__pca_components > self.__datapoints_original.shape[1]:
            raise ValueError(f"pca_components must be between 1 and {self.__datapoints_original.shape[1]}")

        pca = PCA(n_components=self.__pca_components)
        self.__datapoints_pca = pca.fit_transform(self.__datapoints_original)

        # Optionally, you can store the PCA object if you need to transform new data later
        self.__pca = pca

    # Getter methods

    def get_identity(self) -> int:
        """Return the identity (class label) of the Iris Versicolor class."""
        return self.__identity

    def get_features(self) -> list:
        """Return the feature names of the Iris dataset."""
        return self.__features

    def get_datapoints_original(self) -> np.ndarray:
        """Return the original 4D data points of the entire Iris dataset."""
        return self.__datapoints_original

    def get_datapoints_pca(self) -> np.ndarray:
        """Return the PCA-transformed data points of the entire Iris dataset."""
        return self.__datapoints_pca

    def get_versicolor_data(self, transformed: bool = True) -> np.ndarray:
        """
        Extract and return data points belonging to the Iris Versicolor class (label 1).
        Can return either original or PCA-transformed data based on the 'transformed' flag.

        :param transformed: If True, returns PCA-transformed data; otherwise, returns original data
        :return: A numpy array of Iris Versicolor data points
        """
        versicolor_mask = self.__labels == (self.__identity - 1)  # Mask for Iris Versicolor (label 1)

        if transformed:
            return self.__datapoints_pca[versicolor_mask]
        else:
            return self.__datapoints_original[versicolor_mask]

    def get_pca_components(self) -> int:
        """Return the number of PCA components used."""
        return self.__pca_components

    def get_pca_explained_variance_ratio(self) -> np.ndarray:
        """Return the explained variance ratio of the PCA components."""
        return self.__pca.explained_variance_ratio_

    def transform_new_data(self, new_data: np.ndarray) -> np.ndarray:
        """
        Transform new data using the already fitted PCA.

        :param new_data: New data to transform (must have the same number of features as the original data)
        :return: PCA-transformed new data
        """
        return self.__pca.transform(new_data)

    def generate(self, n: int):

        data = self.get_versicolor_data(transformed=True)
        labels = np.full((len(data), 1), self.__identity)
        data = np.concatenate((data, labels), axis=1)

        return data
