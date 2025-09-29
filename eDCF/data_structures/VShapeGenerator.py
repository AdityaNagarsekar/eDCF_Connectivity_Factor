import numpy as np

class VShapeGenerator:
    """
    Creates VShapeGenerator objects that generate V-shaped data points for a specific class,
    with repetitions based on the user input.
    """

    def __init__(self, identity: int, angle: float, shift: float, noise_rate: float, repetitions: int):
        """
        Constructor for VShapeGenerator

        :param identity: Class identity for the line.
        :param angle: Angle of the line in degrees (e.g., 45 or -45).
        :param shift: The shift in the x-axis for each mirrored V.
        :param noise_rate: Noise rate for adding random noise to the data points.
        :param repetitions: Number of V shapes to generate for this class.
        """
        self.__angle = angle  # Angle for the line (either + or - depending on class)
        self.__shift = shift  # Shift in x-axis for each mirrored V shape
        self.__noise_rate = noise_rate  # Noise rate for adding randomness to data points
        self.__identity = identity  # Class identity (label)
        self.__repetitions = repetitions  # Number of times the V shape is repeated

    def generate(self, n: int = 100) -> np.ndarray:
        """
        This method generates 'n' datapoints for each V shape and repeats the process based on repetitions,
        shifting the origin each time.

        :param n: Number of datapoints per line (total will be 2 * n per V shape).
        :return: A numpy array of generated data points.
        """
        all_datapoints = []

        for i in range(self.__repetitions):
            origin_x = i * self.__shift  # Shifted origin for each repetition
            origin_y = 0.0  # Keep y-origin constant

            # Generate datapoints for the line
            slope = np.tan(np.radians(self.__angle))
            x_values = np.linspace(0, 5 if self.__angle > 0 else -5, n) + origin_x
            datapoints = np.zeros((n, 3))

            for j, x in enumerate(x_values):
                y = slope * (x - origin_x) + origin_y
                noise_x = np.random.rand() * self.__noise_rate - self.__noise_rate / 2
                noise_y = np.random.rand() * self.__noise_rate - self.__noise_rate / 2
                datapoints[j] = [x + noise_x, y + noise_y, self.__identity]

            # Append the generated datapoints for this line
            all_datapoints.append(datapoints)

        # Combine all generated datapoints into one array
        all_datapoints = np.vstack(all_datapoints)
        return all_datapoints
