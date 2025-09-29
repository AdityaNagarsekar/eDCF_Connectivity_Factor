import numpy as np

class Mask:
    """
    Loads non-fern data from Mask.npy.
    """

    def __init__(self, identity: int):
        """
        Constructor for MaskData
        """

        self.__datapoints = ...
        self.__identity = identity

    def generate(self, n: int) -> np.ndarray:
        """
        :return: Non-fern datapoints loaded from Mask.npy
        """

        self.__datapoints: np.ndarray = np.load("data_structures/Mask.npy")
        labels = np.full((len(self.__datapoints), 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints
