from typing import Tuple
import numpy as np

class SinusoidalCurve:
    """
    Creates a SinusoidalCurve object with amplitude, phase_difference, y_center, datapoints, identity, and noise_rate as attributes.

    This class provides 'n' number of datapoints on a sinusoidal curve.
    """

    def __init__(self, identity: int, amplitude: float = 1.0, phase_difference: float = 0.0, y_center: float = 0.0, noise_rate: float = 1.0):
        """
        Constructor for SinusoidalCurve

        :param identity: takes the identity of the curve (class label) as an integer (identity > 0)
        :param amplitude: takes the amplitude of the sinusoidal curve
        :param phase_difference: takes the phase difference for the sinusoidal curve
        :param y_center: takes the y-axis offset of the sinusoidal curve
        :param noise_rate: takes the noise rate for adding random noise to the data points
        """
        self.__amplitude = amplitude  # amplitude of the sinusoidal curve
        self.__phase_difference = phase_difference * (np.pi / 180)  # phase shift in radians
        self.__y_center = y_center  # y-axis offset

        self.__datapoints: np.ndarray = ...  # to be initialized later
        self.__identity = identity  # class identity (label)
        self.__noise_rate = noise_rate  # noise rate for adding randomness to data points

    # Getter methods
    def get_identity(self) -> int:
        """Return the identity (class label) of the sinusoidal curve."""
        return self.__identity

    def get_parameters(self) -> Tuple[float, float, float]:
        """Return the amplitude, phase difference, and y_center of the sinusoidal curve."""
        return self.__amplitude, self.__phase_difference, self.__y_center

    def get_datapoints(self) -> np.ndarray:
        """Return the generated data points."""
        return self.__datapoints

    def generate(self, n: int = 100) -> np.ndarray:
        """
        This method generates 'n' datapoints on a sinusoidal curve. The datapoints are stored in the format [x, y, identity].

        :param n: Number of datapoints to generate
        :return: A numpy array of generated data points
        """
        self.__datapoints = np.zeros((n, 3), float)  # initializing datapoints
        x_values = np.linspace(0.0, 2 * np.pi, n)  # generate evenly spaced x values between 0 and 2π

        i: int = 0  # loop control variable

        for x in x_values:
            # Calculate the y-value using the sinusoidal function y = amplitude * sin(x + phase_difference)
            y = self.__amplitude * np.sin(x + self.__phase_difference)

            # Add random noise to x and y coordinates
            noise_x = np.random.rand() * self.__noise_rate - self.__noise_rate / 2
            noise_y = np.random.rand() * self.__noise_rate - self.__noise_rate / 2

            # Assign x, y, and identity/label to the datapoints
            self.__datapoints[i, 0] = x + noise_x  # x-coordinate with noise (fixed at center x = 0.0)
            self.__datapoints[i, 1] = y + noise_y + self.__y_center  # y-coordinate with noise and y-center offset
            self.__datapoints[i, 2] = self.__identity  # identity/label

            i += 1  # increment loop counter

        return self.__datapoints
