from typing import Tuple
import numpy as np
import math


class Sphere4D:
    """
    Creates a Sphere object with radius, center, datapoints, identity and noise_rate as attributes.

    This class has a method generate which generates 'n' number of datapoints on a sphere.
    """

    def __init__(self, identity: int, radius: float = 0.0, center: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0), noise_rate: float = 1.0):
        """
        Constructor for Sphere

        :param identity: takes the identity of the sphere as an integer (identity > 0)
        :param radius: takes the radius of the sphere to be created
        :param center: takes the center of the sphere to be created
        :param noise_rate: takes the noise_rate of the sphere to be created
        """

        self.__radius = radius  # radius assignment
        self.__center = center  # center assignment
        self.__datapoints: np.ndarray = ...  # will be initialized later
        self.__identity = identity  # identity assignment
        self.__noise_rate = noise_rate  # noise rate assignment

    # Getter methods

    def get_identity(self) -> int:
        """

        :return: identity of the sphere
        """

        return self.__identity

    def get_radius(self) -> float:
        """

        :return: radius of the sphere
        """

        return self.__radius

    def get_center(self) -> Tuple[float, float, float, float]:
        """

        :return: center of the sphere
        """

        return self.__center

    def get_datapoints(self) -> np.ndarray:
        """
        This method requires generate to be called at least once before.

        :return: datapoints created
        """

        return self.__datapoints

    # Getters end

    def generate(self, n: int = 100) -> np.ndarray:
        """
        This method assigns datapoints which are randomly spaced in terms of spherical coordinates.

        The datapoints are assigned in the format [x, y, z, identity] where x, y, and z are the coordinates
        and identity is the label of the point.

        :param n: takes the number of datapoints to be created
        :return: datapoints generated
        """

        self.__datapoints = np.zeros((n, 5), float)  # initializing datapoints

        # Generate random angles and noise
        theta = np.random.uniform(0, 360, n)  # azimuthal angle in degrees
        phi = np.random.uniform(0, 180, n)  # polar angle in degrees
        zeta = np.random.uniform(0, 180, n)  # polar angle in degrees


        i: int = 0  # loop control variable

        for t, p, z in zip(theta, phi, zeta):  # assigning loop

            noise_x = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing x coordinate noise
            noise_y = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing y coordinate noise
            noise_z = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing z coordinate noise
            noise_w = np.random.rand() * self.__noise_rate + np.random.rand() * -self.__noise_rate  # Implementing w coordinate noise

            # Handle cases where theta or phi is exactly 90 or 270 degrees
            if t == 90 or t == 270:
                # If theta is 90 or 270, cos(theta) is 0, making x = 0; handle precision issues manually
                x = 0 + noise_x
            else:
                x = self.__radius * math.sin(math.radians(p)) * math.sin(math.radians(z)) * math.cos(math.radians(t)) + noise_x

            if p == 90:
                # If phi is 90 or 270, z = 0, handle precision manually
                z = 0 + noise_z
            else:
                z = self.__radius * math.sin(math.radians(z)) * math.cos(math.radians(p)) + noise_z

            if z == 90:
                # If phi is 90 or 270, z = 0, handle precision manually
                w = 0 + noise_w
            else:
                w = self.__radius * math.cos(math.radians(z)) + noise_w

            # For y, continue the general case as sin(theta) is well-behaved for all values
            y = self.__radius * math.sin(math.radians(p)) * math.sin(math.radians(t)) + noise_y

            self.__datapoints[i, 0] = x + self.__center[0]  # x assignment
            self.__datapoints[i, 1] = y + self.__center[1]  # y assignment
            self.__datapoints[i, 2] = z + self.__center[2]  # z assignment
            self.__datapoints[i, 3] = w + self.__center[3]  # w assignment
            self.__datapoints[i, 4] = self.__identity  # identity/label assignment

            i += 1  # loop update

        return self.__datapoints
