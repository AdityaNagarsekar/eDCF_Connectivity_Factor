# import statements
from typing import Tuple
import numpy as np
import math


class Circle:
    """
    Creates a Circle object with radius, center, datapoints, identity and noise_rate as attributes.

    This class has a method assign_datapoints which must be implemented before calling get_datapoints.

    The purpose of this class is to provide 'n' number of datapoints on a circle.
    """

    def __init__(self, identity: int, radius: float = 0.0, center: Tuple[float, float] = (0.0, 0.0), noise_rate: float = 1.0, filled_in: bool = False):
        """
        Constructor for Circle

        :param identity: takes the identity of the circle as an integer (identity > 0)
        :param radius: takes the radius of the circle to be created
        :param center: takes the center of the circle to be created
        :param noise_rate: takes the noise_rate of the circle to be created
        :param filled_in: constructs a filled in circle if true.
        """

        self.__radius = radius  # radius assignment
        self.__center = center  # center assignment
        self.__datapoints: np.ndarray = ...  # will be initialized later
        self.__identity = identity  # identity assignment
        self.__noise_rate = noise_rate  # noise rate assignment
        self.__filled_in: bool = filled_in

    # Getter methods

    def get_identity(self) -> int:
        """

        :return: identity of the circle
        """

        return self.__identity

    def get_radius(self) -> float:
        """

        :return: radius of the circle
        """

        return self.__radius

    def get_center(self) -> Tuple[float, float]:
        """

        :return: center of the circle
        """

        return self.__center

    def get_datapoints(self) -> np.ndarray:
        """
        This method requires assign_datapoints to be called at least once before.

        :return: datapoints created
        """

        return self.__datapoints

    # Getters end

    def generate(self, n: int = 100) -> np.ndarray:
        """
        This method assigns datapoints which are randomly spaced in terms of angle theta.

        The datapoints are assigned in the format [x, y, identity] where x is the x coordinate y is the y coordinate and
        identity is the label of the point.

        :param n: takes the number of datapoints to be created

        :return: None
        """

        self.__datapoints = np.zeros((n, 3), float)  # initializing datapoints

        radii = np.sqrt(np.random.uniform(0, 1, n)) * self.__radius  # Uniform distribution within the circle
        theta = np.random.uniform(0, 360, n)  # Generating random angles in degree

        for i in range(len(theta)):  # assigning loop

            noise_x = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing x coordinate noise
            noise_y = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate # Implementing y coordinate noise

            # since cos(90) and cos(270) are not 0 in math.cos we have to set them.
            if theta[i] == 90.0 or theta[i] == 270.0:

                if self.__filled_in:
                    self.__datapoints[i, 0] = self.__center[0] + noise_x  # x assignment
                    self.__datapoints[i, 1] = radii[i] * math.sin(math.radians(theta[i])) + self.__center[1] + noise_y  # y assignment
                    self.__datapoints[i, 2] = self.__identity  # identity/label assignment
                else:
                    self.__datapoints[i, 0] = self.__center[0] + noise_x  # x assignment
                    self.__datapoints[i, 1] = radii[i] * math.sin(math.radians(theta[i])) + self.__center[1] + noise_y  # y assignment
                    self.__datapoints[i, 2] = self.__identity  # identity/label assignment

            # general assignment
            else:

                if self.__filled_in:
                    self.__datapoints[i, 0] = radii[i] * math.cos(math.radians(theta[i])) + self.__center[0] + noise_x  # x assignment
                    self.__datapoints[i, 1] = radii[i] * math.sin(math.radians(theta[i])) + self.__center[1] + noise_y  # y assignment
                    self.__datapoints[i, 2] = self.__identity  # identity/label assignment
                else:
                    self.__datapoints[i, 0] = self.__radius * math.cos(math.radians(theta[i])) + self.__center[0] + noise_x  # x assignment
                    self.__datapoints[i, 1] = self.__radius * math.sin(math.radians(theta[i])) + self.__center[1] + noise_y  # y assignment
                    self.__datapoints[i, 2] = self.__identity  # identity/label assignment

            # Uses the formula x(p) = r.cos(a) + x(c) and y(p) = r.sin(a) + y(c)

            i += 1  # loop update

        return self.__datapoints
