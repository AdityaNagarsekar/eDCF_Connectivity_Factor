import numpy as np

class Weight:
    """
    Weight manager for multi-scale triangular basis functions.

    Manages weight sums across scales using triangular ("hat") functions.

    Attributes
    ----------
    dim : int
        Number of scales (dimensionality).
    weights : dict[int, float]
        Accumulated weight per scale index.
    sets : numpy.ndarray, shape (dim+1, 3)
        Boundaries [left, center, right] for each triangular function.

    Methods
    -------
    linear_gen(x, center, left, right) -> float
        Evaluate triangular hat function at x.
    add_point_info(x) -> None
        Accumulate weight contributions for input x.
    __add__(other) -> None
        In-place addition of another Weight's weights.

    Examples
    --------
    >>> w = Weight(dim=2)
    >>> w.add_point_info(1.5)
    >>> print(w.weights)
    """

    def __init__(self, dim: int):
        """
        Initialize Weight object with given dimensionality.

        Parameters
        ----------
        dim : int
            Number of scales (dimensionality) for triangular functions.

        Notes
        -----
        Initializes:
        - `weights`: zeros for each scale index.
        - `sets`: array defining triangular function boundaries.
        """

        self.dim = dim

        self.weights = {}

        for i in range(dim + 1):
            self.weights[i] = 0.0

        # Calculate bounds based on 3^m - 1. Example: dim=2 -> bounds=[0, 2, 8]
        bounds = [3 ** m - 1 for m in range(dim + 1)]

        # Create the 'sets' array to store [left, center, right] for each function
        # Initialize with -1 (float type for compatibility with function)
        self.sets = np.full((len(bounds), 3), -1.0)  # Use -1.0 for float

        for i in range(dim + 1):
            # Center is the current bound
            self.sets[i, 1] = float(bounds[i])

            # Left boundary is the previous bound (if it exists)
            if i != 0:
                self.sets[i, 0] = float(bounds[i - 1])

            # Right boundary is the next bound (if it exists)
            if i != dim:
                self.sets[i, 2] = float(bounds[i + 1])

    @staticmethod
    def linear_gen(x: float, center: float, left: float, right: float):
        """
        Triangular (hat) basis function.

        Ramps from 0 at `left` to 1 at `center`, then back to 0 at `right`.

        Parameters
        ----------
        x : float
            Value at which to evaluate the function.
        center : float
            Peak position.
        left : float
            Left boundary (value 0) or -1 for half-triangle.
        right : float
            Right boundary (value 0) or -1 for half-triangle.

        Returns
        -------
        float
            Function value between 0 and 1.

        Notes
        -----
        - If `left == -1`, uses descending slope from `center` to `right`.
        - If `right == -1`, uses ascending slope from `left` to `center`.
        """
        influence: float = 0.0 # Initialize influence to 0.0

        # --- Input Validation and Handling Edge Cases ---
        # Check for invalid interval where boundaries are defined but center isn't between them
        if left != -1 and right != -1 and not (left <= center <= right):
            # Or perhaps raise ValueError("Center must be between left and right")
            return 0.0 # If center is outside defined bounds, influence is 0

        # Check for zero-width intervals which cause division by zero
        if left != -1 and abs(center - left) < 1e-9 and left <= x <= center:
             # If x is exactly at the center==left peak, define influence as 1
             return 1.0 if x == center else 0.0
        if right != -1 and abs(right - center) < 1e-9 and center <= x <= right:
             # If x is exactly at the center==right peak, define influence as 1
             return 1.0 if x == center else 0.0
        # Avoid division by zero if left or right boundary is defined but equals center
        if left != -1 and abs(center - left) < 1e-9:
          pass # Calculation below will handle this via x conditions
        if right != -1 and abs(right - center) < 1e-9:
          pass # Calculation below will handle this via x conditions

        # --- Core Logic for Different Cases ---
        if left == -1 and right == -1:
            # This case shouldn't happen if bounds are generated correctly, but handle it.
            # Represents a single point impulse at 'center'? Or undefined? Let's return 0.
             # raise ValueError("Invalid left == -1 and right == -1.") # Or handle differently
             return 0.0

        elif left == -1:
            # Right half-triangle (or line segment descending from center)
            # Ensure right is actually to the right of center
            if right <= center:
                 # raise ValueError("Invalid configuration: left=-1 but right <= center")
                 return 0.0 # Undefined or invalid setup
            if center <= x <= right:
                # Avoid division by zero checked above
                influence = (right - x) / (right - center)

        elif right == -1:
            # Left half-triangle (or line segment ascending to center)
            # Ensure left is actually to the left of center
            if left >= center:
                 # raise ValueError("Invalid configuration: right=-1 but left >= center")
                 return 0.0 # Undefined or invalid setup
            if left <= x <= center:
                # Avoid division by zero checked above
                influence = (x - left) / (center - left)

        else:
            # Full triangle function defined over [left, right] peaking at center
            if left <= x <= center:
                 # Avoid division by zero checked above
                influence = (x - left) / (center - left)
            elif center < x <= right:
                 # Avoid division by zero checked above
                influence = (right - x) / (right - center)
            # Outside [left, right], influence remains 0

        # Ensure influence doesn't go slightly out of [0, 1] due to float precision
        influence = max(0.0, min(1.0, influence))

        return influence

    def add_point_info(self, x: float | int):
        """
        Accumulate weight contributions for a given input value.

        Parameters
        ----------
        x : float or int
            Input value to evaluate against triangular functions.

        Returns
        -------
        None
            Updates `self.weights` in-place.
        """

        for i in range(self.dim + 1):

            self.weights[i] += Weight.linear_gen(x, float(self.sets[i, 1]), float(self.sets[i, 0]), float(self.sets[i, 2]))

    def __add__(self, other):
        """
        In-place addition of another Weight's weights.

        Parameters
        ----------
        other : Weight
            Weight instance to add.

        Returns
        -------
        None
            Modifies `self.weights` in-place.
        """

        for i in range(self.dim + 1):
            self.weights[i] += other.weights[i]
