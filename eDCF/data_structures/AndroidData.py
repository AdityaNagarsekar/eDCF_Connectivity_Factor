import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class AndroidData:

    def __init__(self, identity: int):
        """
        Constructor for AndroidData
        """

        self.__file_path = 'DatapointsStorage/AndroidData/ratings_Apps_for_Android.csv'  # Update with your file path
        self.__identity = identity
        self.__datapoints = ...

    def generate(self, n: int) -> np.ndarray:

        data = pd.read_csv(self.__file_path)

        # Initialize LabelEncoders for the alphanumeric columns
        label_encoders = {}
        for col in data.columns[:2]:  # Assuming the first two columns are alphanumeric
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])  # Encode the column
            label_encoders[col] = le  # Store the encoder for future use

        # Convert the data into a NumPy array
        numerical_data = data.to_numpy()

        self.__datapoints = numerical_data

        labels = np.full((len(self.__datapoints), 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints
