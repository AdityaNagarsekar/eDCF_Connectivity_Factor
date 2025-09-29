# import statements
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    """
    Provide utilities for hyperparameter tuning, training, and prediction using
    a K-Nearest Neighbors classifier.

    Parameters
    ----------
    k_start : int
        Lower bound for 'k' (number of neighbors) in grid search.
    p_start : int
        Lower bound for Minkowski distance power parameter.
    p_lim : int
        Upper bound for Minkowski distance power parameter.
    k_lim : int
        Upper bound for 'k' in grid search.
    cvn : int
        Number of folds for cross-validation.
    leaf_start : int
        Lower bound for leaf size in grid search.
    leaf_lim : int
        Upper bound for leaf size in grid search.

    Attributes
    ----------
    best_knn : KNeighborsClassifier
        KNN model fitted with best hyperparameters.
    best_params : dict
        Best hyperparameters found by grid search.

    Methods
    -------
    _calculate_k_lim() -> int
        Determine the maximum viable 'k' based on training set size.
    train_hyper(core_ctrl: int = -1) -> None
        Perform GridSearchCV tuning and fit the KNN classifier.
    initialize_hyper() -> None
        Load saved hyperparameters and refit the classifier.
    execute(grid_point: array-like) -> float
        Predict the class label for a single data point.

    Examples
    --------
    >>> knn = KNN(1, 2, 5, 10, 5, 1, 5)
    >>> knn.train_hyper()
    >>> label = knn.execute([x1, x2, x3])
    """

    def __init__(self, k_start: int, p_start: int, p_lim: int, k_lim: int, cvn: int, leaf_start: int, leaf_lim: int):
        """
        Initialize KNN with specified hyperparameter search ranges.

        Parameters
        ----------
        k_start : int
            Lower bound for number of neighbors (k) in grid search.
        p_start : int
            Lower bound for Minkowski distance power parameter.
        p_lim : int
            Upper bound for Minkowski power parameter.
        k_lim : int
            Upper bound for number of neighbors in grid search.
        cvn : int
            Number of cross‑validation folds.
        leaf_start : int
            Lower bound for leaf size in grid search.
        leaf_lim : int
            Upper bound for leaf size in grid search.
        """

        self.k_start: int = k_start  # k_start initialize
        self.p_start: int = p_start  # p_start initialize
        self.p_lim: int = p_lim  # p_lim initialize
        self.k_lim: int = k_lim  # k_lim initialize

        # ..........
        # set __k_lim to -1 to get the maximum k limit
        # ..........

        self.cvn: int = cvn  # initialize cvn (cross-validation folds)
        self.leaf_start: int = leaf_start  # leaf_start initialize
        self.leaf_lim: int = leaf_lim  # leaf_lim initialize

        self.X_test = ...  # Placeholder for test features
        self.X_train = ...  # Placeholder for training features
        self.y_test = ...  # Placeholder for test labels
        self.y_train = ...  # Placeholder for training labels
        self.hyper_params = ...  # Placeholder for best hyperparameters
        self.best_knn = ...  # Placeholder for the best KNN model

    def __calculate_k_lim(self) -> None:
        """
        Determine and set the maximum 'k' (neighbors) based on dataset size.

        Loads data from 'Datapoints.npy' to assess total samples, then:
        - If `self.k_lim` is -1, uses total sample count.
        - Otherwise, sets `self.k_lim = min(self.k_lim, total_samples)`.

        Returns
        -------
        None
            Updates the `self.k_lim` attribute in-place.
        """

        # Load the data from 'Datapoints.npy'
        datapoints = np.load("Datapoints.npy", allow_pickle=True)

        # Get the length of the training data
        max_val: int = 0

        for i in range(len(datapoints)):
            X_train_length = int(len(np.load("Datapoints.npy", allow_pickle=True)[0]) * 0.8) // len(datapoints)
            if max_val < X_train_length:
                max_val = X_train_length

        X_train_length = max_val
        # Check if the manual k limit is too high for the system
        if X_train_length < self.k_lim // len(datapoints):
            self.k_lim = len(datapoints) * X_train_length

        # Automatically adjust k limit
        if self.k_lim == -1:
            self.k_lim = len(datapoints) * X_train_length
        
        return None

    def train_hyper(self, core_ctrl: int = -1) -> None:
        """
        Tune and evaluate the KNN classifier using GridSearchCV.

        Workflow:
        1. Load 'Datapoints.npy' and stack samples into array.
        2. Split into features (X) and labels (y).
        3. Perform an 80/20 train/test split.
        4. Adjust `k_lim` via `self.__calculate_k_lim()`.
        5. Define parameter grid for `n_neighbors`, `p`, and `leaf_size`.
        6. Execute GridSearchCV with `self.cvn` folds and `core_ctrl` parallel jobs.
        7. Store the best estimator in `self.best_knn` and best params in `self.hyper_params`.

        Parameters
        ----------
        core_ctrl : int, optional
            Number of parallel jobs for GridSearchCV (default: -1, uses all cores).

        Returns
        -------
        None
        """

        # Load the data from 'Datapoints.npy'
        data = np.load('Datapoints.npy', allow_pickle=True)
        data = np.vstack(data)

        # Split data into features (X) and labels (y), reshaping it for compatibility with sklearn models
        X = data[:, :-1]
        y = data[:, -1].astype(int).reshape(len(data))

        # Split data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the KNN classifier
        knn = KNeighborsClassifier()

        # Calculate the k limit (number of neighbors)
        self.__calculate_k_lim()

        # Define the grid of hyperparameters to search
        param_grid = {
            'n_neighbors': np.arange(self.k_start, self.k_lim, 2),
            'p': np.arange(self.p_start, self.p_lim),
            'leaf_size': np.arange(self.leaf_start, self.leaf_lim)
        }

        # Initialize KFold cross-validation
        cv = KFold(n_splits=self.cvn, shuffle=True, random_state=42)

        # Initialize GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='f1_macro', n_jobs=core_ctrl)

        # Fit the model using the grid search
        grid_search.fit(X, y)

        # Store the best hyperparameters and model
        hyper_params = grid_search.best_params_
        best_knn = grid_search.best_estimator_

        # Evaluate the best model on the test data
        f1_score = best_knn.score(X_test, y_test)

        # Print the best hyperparameters and F1 score
        print(hyper_params)
        print(f"F1 Score on Testing Data : {f1_score}")

        # Save F1 score and hyperparameters to files
        f1_score = np.asarray(f1_score)
        np.save("algorithms/Testing_Score", f1_score)
        np.save("algorithms/Hyper_Param", hyper_params)

        # Save the train and test sets for future use
        np.save('algorithms/X_train', X_train)
        np.save('algorithms/y_train', y_train)
        np.save('algorithms/X_test', X_test)
        np.save('algorithms/y_test', y_test)

        return None

    def initialize_hyper(self) -> None:
        """
        Load saved hyperparameters and retrain the KNN classifier.

        Workflow:
        1. Load hyperparameters dict from 'algorithms/Hyper_Param.npy' into `self.hyper_params`.
        2. Initialize `self.best_knn` using `self.hyper_params`.
        3. Load training data from 'algorithms/X_train.npy' and 'algorithms/y_train.npy'.
        4. Fit `self.best_knn` on the full training set (`X_train`, `y_train`).

        Returns
        -------
        None
        """

        # Load the best hyperparameters from file
        self.hyper_params = np.load("algorithms/Hyper_Param.npy", allow_pickle=True).item()

        # Initialize the KNN classifier with the loaded hyperparameters
        self.best_knn = KNeighborsClassifier(**self.hyper_params)

        # Load the training data
        self.X_train = np.load("algorithms/X_train.npy", allow_pickle=True)
        self.y_train = np.load('algorithms/y_train.npy', allow_pickle=True)

        # Fit the model on the entire training set
        self.best_knn.fit(self.X_train, self.y_train)

        return None

    def execute(self, grid_point) -> float:
        """
        Predict the class label for a single data point using the trained KNN classifier.

        Parameters
        ----------
        grid_point : array-like
            1D sequence of feature values for prediction. Ignores last element as it represents a label.

        Returns
        -------
        float
            Predicted class label.
        """
        
        # Extract the feature values (excluding the label)
        data = np.array(grid_point[:len(grid_point) - 1])

        # Reshape the data into 2D array format (1, n_features)
        data = data.reshape(1, -1)

        # Predict the label using the trained KNN classifier
        prediction = self.best_knn.predict(data)

        return float(prediction[0])  # Return the predicted label as a float
