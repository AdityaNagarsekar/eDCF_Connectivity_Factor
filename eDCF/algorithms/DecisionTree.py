# import statements
import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


class DecisionTree:
    """
    Provide functionality to train and apply a Decision Tree classifier with hyperparameter tuning.

    Parameters
    ----------
    criterion : list[str]
        Candidate split criteria (e.g., 'gini', 'entropy').
    splitter : list[str]
        Candidate splitter strategies ('best', 'random').
    max_depth : list[int]
        Candidate maximum tree depths.
    min_samples_split : list[int]
        Candidate minimum samples required to split a node.
    min_samples_leaf : list[int]
        Candidate minimum samples required at a leaf node.
    max_features : list[int|float|str]
        Candidate number of features to consider at each split.

    Attributes
    ----------
    best_dtc : DecisionTreeClassifier
        Classifier selected by grid search.
    best_params : dict
        Best hyperparameters found during tuning.
    X_train, X_test : np.ndarray
        Training and testing feature sets.
    y_train, y_test : np.ndarray
        Training and testing labels.

    Methods
    -------
    train_hyper(core_ctrl: int = -1) -> None
        Run GridSearchCV to tune hyperparameters and fit the classifier.
    train() -> None
        Train the classifier on the full training set with best hyperparameters.
    initialize_hyper() -> None
        Load saved hyperparameters and retrain the model.
    execute(grid_point: np.ndarray) -> int
        Predict the label for a single data point.

    Examples
    --------
    >>> dt = DecisionTree(
    ...     ['gini', 'entropy'], ['best'], [10, None], [2, 5], [1], ['sqrt']
    >>> dt.train_hyper()
    >>> label = dt.execute(np.array([x1, x2, x3]))
    """

    def __init__(self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features):
        """
        Initialize DecisionTree with hyperparameter search space.

        Parameters
        ----------
        criterion : list[str]
            Candidate split criteria for GridSearchCV (e.g., ['gini', 'entropy']).
        splitter : list[str]
            Candidate splitter algorithms (e.g., ['best', 'random']).
        max_depth : list[int or None]
            Candidate maximum tree depths (None for no limit).
        min_samples_split : list[int]
            Candidate minimum samples required to split an internal node.
        min_samples_leaf : list[int]
            Candidate minimum samples required to be at a leaf node.
        max_features : list[int, float, or str]
            Candidate number of features to consider at each split (e.g., ['sqrt', 0.8]).
        """

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.best_dtc = ...  # Placeholder for the best Decision Tree model
        self.best_params = ...  # Placeholder for the best hyperparameters
        self.X_test = ...  # Placeholder for test features
        self.X_train = ...  # Placeholder for training features
        self.y_test = ...  # Placeholder for test labels
        self.y_train = ...  # Placeholder for training labels

    def train_hyper(self, core_ctrl: int = -1) -> None:
        """
        Tune and fit the Decision Tree classifier using GridSearchCV and cross-validation.

        Workflow:
        1. Load data from 'Datapoints.npy'.
        2. Split into 80% training and 20% testing sets.
        3. Perform grid search over hyperparameters (criterion, splitter, max_depth, etc.)
           using KFold cross-validation.
        4. Store the best estimator in `self.best_dtc` and parameters in `self.best_params`.

        Parameters
        ----------
        core_ctrl : int, optional
            Number of parallel jobs for GridSearchCV. -1 uses all available CPU cores.

        Returns
        -------
        None
        """

        # Load data from the 'Datapoints.npy' file
        data = np.load('Datapoints.npy', allow_pickle=True)

        data = np.vstack(data)

        # Split data into features (X) and labels (y), reshaping it for compatibility with sklearn models
        X = data[:, :-1]
        y = data[:, -1].astype(int).reshape(len(data))

        # Split the data into training and testing sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the grid of hyperparameters to search through
        parameter_space = {
            'criterion': self.criterion,
            'splitter': self.splitter,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
        }

        # Initialize the Decision Tree classifier
        dtc = DecisionTreeClassifier(random_state=42)

        # Define KFold cross-validation with 5 splits
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize GridSearchCV for hyperparameter tuning
        clf = GridSearchCV(dtc, parameter_space, cv=cv, scoring='f1_macro', n_jobs=core_ctrl)

        # Fit the model using training data
        clf.fit(self.X_train, self.y_train)

        # Print the best hyperparameters and cross-validation score
        print('Best parameters found:\n', clf.best_params_)
        print("Best cross-validation score:", clf.best_score_)

        # Evaluate the model on the test data
        test_score = clf.score(self.X_test, self.y_test)
        print("Test accuracy before retraining (scikit-learn):", test_score)

        # Save the test score and best hyperparameters to files
        np.save("algorithms/Testing_Score", np.asarray(test_score))
        np.save('algorithms/Hyper_Param', clf.best_params_)

        # Save the train and test sets to files for later use
        np.save('algorithms/X_train', self.X_train)
        np.save('algorithms/y_train', self.y_train)
        np.save('algorithms/X_test', self.X_test)
        np.save('algorithms/y_test', self.y_test)

        return None

    def train(self) -> None:
        """
        Train the Decision Tree on the full training set using best hyperparameters.

        Workflow:
        1. Load `X_train` and `y_train` from previously saved NumPy files ('algorithms/X_train.npy', 'algorithms/y_train.npy').
        2. Initialize `DecisionTreeClassifier` with `self.best_params`.
        3. Fit the classifier on `X_train` and `y_train`.

        Returns
        -------
        None
        """

        # Load training data from previously saved files
        self.X_train = np.load('algorithms/X_train.npy', allow_pickle=True)
        self.y_train = np.load('algorithms/y_train.npy', allow_pickle=True)

        # Initialize the DecisionTreeClassifier with the best hyperparameters
        self.best_dtc = DecisionTreeClassifier(**self.best_params)

        # Fit the model on the entire training set
        self.best_dtc.fit(self.X_train, self.y_train)

        return None

    def initialize_hyper(self) -> None:
        """
        Load saved hyperparameters and retrain the Decision Tree classifier.

        Workflow:
        1. Load hyperparameters dict from 'algorithms/Hyper_Param.npy'.
        2. Update `self.best_params`.
        3. Invoke `self.train()` to fit the classifier using loaded hyperparameters.

        Returns
        -------
        None
        """

        # Load the best hyperparameters from a file
        hyper_params = np.load("algorithms/Hyper_Param.npy", allow_pickle=True)
        self.best_params = hyper_params.item()  # Extract the hyperparameters as a dictionary
        self.train()  # Train the model with the best hyperparameters

        return None

    def execute(self, grid_point) -> float:
        """
        Predict the class label for a single data point using the trained classifier.

        Parameters
        ----------
        grid_point : array-like
            1D sequence of feature values for prediction. Label is ignored being the last index of a datapoint.

        Returns
        -------
        float
            Predicted class label.
        """
        
        # Extract features (ignoring the label if present)
        features = np.array(grid_point[:len(grid_point) - 1])

        # Reshape to 2D array with shape (1, n_features)
        features = features.reshape(1, -1)

        # Use the trained model to predict the label
        prediction = self.best_dtc.predict(features)

        return float(prediction[0])  # Return the predicted label as a float
