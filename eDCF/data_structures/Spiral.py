# import statements
from typing import Tuple
import numpy as np
import math


class Spiral:
    """
    Creates a Spiral object with radius, center, datapoints, identity and noise_rate as attributes.

    This class has a method assign_datapoints which must be implemented before calling get_datapoints.

    The purpose of this class is to provide 'n' number of datapoints on a spiral.
    """

    def __init__(self, identity: int, angle_start: int = 0, angle_end: int = 0, center: Tuple[float, float] = (0.0, 0.0), noise_rate: float = 1.0):
        """
        Constructor for Spiral

        :param identity: takes the identity of the spiral as an integer (identity > 0)
        :param angle_start: takes the angle of the spiral to be created
        :param center: takes the center of the spiral to be created
        :param noise_rate: takes the noise_rate of the spiral to be created
        """

        self.__angle_start = angle_start  # angle start assignment
        self.__angle_end = angle_end  # angle end assignment
        self.__center = center  # center assignment
        self.__datapoints: np.ndarray = ...  # will be initialized later
        self.__identity = identity  # identity assignment
        self.__noise_rate = noise_rate  # noise rate assignment

    # Getter methods

    def get_identity(self) -> int:
        """

        :return: identity of the spiral
        """

        return self.__identity

    def get_angle(self) -> Tuple[float, float]:
        """

        :return: radius of the spiral
        """

        return self.__angle_start, self.__angle_end

    def get_center(self) -> Tuple[float, float]:
        """

        :return: center of the spiral
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

        theta = np.random.uniform(self.__angle_start, self.__angle_end, n)  # Generating random angles in degree

        i: int = 0  # loop control variable

        for phi in theta:  # assigning loop

            noise_x = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing x coordinate noise
            noise_y = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate # Implementing y coordinate noise

            # since cos(90) and cos(270) are not 0 in math.cos we have to set them.
            if phi == 90.0 or phi == 270.0:
                self.__datapoints[i, 0] = self.__center[0] + noise_x  # x assignment
                self.__datapoints[i, 1] = np.abs(math.radians(phi) - math.radians(self.__angle_start)) * math.sin(math.radians(phi)) + self.__center[1] + noise_y  # y assignment
                self.__datapoints[i, 2] = self.__identity  # identity/label assignment

            # general assignment
            else:
                self.__datapoints[i, 0] = np.abs(math.radians(phi) - math.radians(self.__angle_start)) * math.cos(math.radians(phi)) + self.__center[0] + noise_x  # x assignment
                self.__datapoints[i, 1] = np.abs(math.radians(phi) - math.radians(self.__angle_start)) * math.sin(math.radians(phi)) + self.__center[1] + noise_y  # y assignment
                self.__datapoints[i, 2] = self.__identity  # identity/label assignment

            # Uses the formula x(p) = r.cos(a) + x(c) and y(p) = r.sin(a) + y(c)

            i += 1  # loop update

        return self.__datapoints
