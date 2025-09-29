# import statements
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


class MLP:
    """
    Provide functionality to train, evaluate, and predict using a Multilayer Perceptron (MLP) classifier
    with hyperparameter tuning via scikit-learn's MLPClassifier and GridSearchCV.

    Parameters
    ----------
    hidden_layer_sizes : list[tuple[int, ...]]
        Candidate hidden layer configurations for grid search.
    activation : list[str]
        Activation functions to consider (e.g., 'relu', 'tanh').
    solver : list[str]
        Solvers for weight optimization (e.g., 'adam', 'sgd').
    alpha : list[float]
        L2 penalty (regularization) values to search.
    learning_rate : list[str]
        Learning rate schedules to consider (e.g., 'constant', 'adaptive').
    max_iter : list[int]
        Maximum number of iterations values to search.

    Attributes
    ----------
    best_mlp : MLPClassifier
        Fitted MLP classifier with optimal hyperparameters.
    best_params : dict
        Best hyperparameter values found by GridSearchCV.
    X_train : np.ndarray
        Training set features.
    X_test : np.ndarray
        Test set features.
    y_train : np.ndarray
        Training set labels.
    y_test : np.ndarray
        Test set labels.

    Methods
    -------
    train_hyper(core_ctrl: int = -1) -> None
        Perform hyperparameter tuning via GridSearchCV and fit the MLP.
    train() -> None
        Train the MLP on the full training set using best hyperparameters.
    initialize_hyper() -> None
        Load saved hyperparameters and retrain the MLP classifier.
    execute(grid_point: array-like) -> float
        Predict class label for a single data point.

    Examples
    --------
    >>> mlp = MLP([(50,), (100,)], ['relu'], ['adam'], [1e-4], ['constant'], [200])
    >>> mlp.train_hyper()
    >>> label = mlp.execute([x1, x2, x3])
    """

    def __init__(self, hidden_layer_sizes, activation, solver, alpha, learning_rate, max_iter):
        """
        Initialize MLP with specified hyperparameter search space.

        Parameters
        ----------
        hidden_layer_sizes : list[tuple[int, ...]]
            Candidate configurations for the number and size of hidden layers.
        activation : list[str]
            Activation functions to consider (e.g., ['relu', 'tanh']).
        solver : list[str]
            Solvers for weight optimization (e.g., ['adam', 'sgd']).
        alpha : list[float]
            L2 penalty (regularization strength) values to search.
        learning_rate : list[str]
            Learning rate schedules to consider (e.g., ['constant', 'adaptive']).
        max_iter : list[int]
            Maximum number of training iterations to consider.
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.best_mlp = ...  # To store the best model found
        self.best_params = ...  # To store the best hyperparameters
        self.X_test = ...  # To store the test set features
        self.X_train = ...  # To store the training set features
        self.y_test = ...  # To store the test set labels
        self.y_train = ...  # To store the training set labels

    def train_hyper(self, core_ctrl: int = -1) -> None:
        """
        Tune and evaluate the MLP classifier using GridSearchCV.

        Workflow:
        1. Load 'Datapoints.npy' and stack into array.
        2. Split into features (X) and labels (y).
        3. Perform 80/20 train/test split.
        4. Define parameter grid for hidden_layer_sizes, activation, solver, alpha, learning_rate, max_iter.
        5. Execute GridSearchCV with `self.cvn` folds and `core_ctrl` parallel jobs.
        6. Store best estimator in `self.best_mlp` and parameters in `self.best_params`.
        7. Save split datasets and best parameters for future use.

        Parameters
        ----------
        core_ctrl : int, optional
            Number of parallel jobs for GridSearchCV (default: -1 uses all CPUs).

        Returns
        -------
        None
        """

        # Load data from 'Datapoints.npy'
        data = np.load('Datapoints.npy', allow_pickle=True)
        data = np.vstack(data)

        # Split data into features (X) and labels (y)
        X = data[:, :-1]
        y = data[:, -1].astype(int).reshape(len(data))

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the parameter grid to search
        parameter_space = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter  # Set a high max_iter to ensure convergence
        }

        # Initialize the MLPClassifier
        mlp = MLPClassifier()

        # Define k-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize GridSearchCV
        clf = GridSearchCV(mlp, parameter_space, cv=cv, scoring='f1_macro', n_jobs=core_ctrl)

        # Fit the model
        clf.fit(self.X_train, self.y_train)

        # Print best parameters and score
        print('Best parameters found:\n', clf.best_params_)
        print("Best cross-validation score:", clf.best_score_)

        # Evaluate on test data using the best estimator found
        test_score = clf.score(self.X_test, self.y_test)
        print("Test accuracy before retraining (scikit-learn):", test_score)

        test_score = np.asarray(test_score)

        # Save test score and best hyperparameters to files
        np.save("algorithms/Testing_Score", test_score)
        np.save('algorithms/Hyper_Param', clf.best_params_)

        # Save the train-test split data
        np.save('algorithms/X_train', self.X_train)
        np.save('algorithms/y_train', self.y_train)
        np.save('algorithms/X_test', self.X_test)
        np.save('algorithms/y_test', self.y_test)

        return None

    def train(self):
        """
        Retrain the MLP model using optimal hyperparameters.

        Workflow:
        1. Initialize MLPClassifier with `self.best_params`.
        2. Load training data from 'algorithms/X_train.npy' and 'algorithms/y_train.npy'.
        3. Fit the classifier on the entire training set.

        Returns
        -------
        None
        """

        # Retrain the model using the best hyperparameters on the entire training set
        self.best_mlp = MLPClassifier(**self.best_params)

        # Load previously saved training data
        self.X_train = np.load("algorithms/X_train.npy", allow_pickle=True)
        self.y_train = np.load('algorithms/y_train.npy', allow_pickle=True)

        # Fit the model on the entire training data
        self.best_mlp.fit(self.X_train, self.y_train)

        return None

    def initialize_hyper(self):
        """
        Load saved hyperparameters and retrain the MLP classifier.

        Workflow:
        1. Load hyperparameters from 'algorithms/Hyper_Param.npy' into `self.best_params`.
        2. Call `self.train()` to fit the model on the full training set.

        Returns
        -------
        None
        """

        # Load best hyperparameters
        hyper_params = np.load("algorithms/Hyper_Param.npy", allow_pickle=True)
        self.best_params = hyper_params.item()

        # Retrain the model using the loaded hyperparameters
        self.train()

        return None

    def execute(self, grid_point):
        """
        Predict the class label for a single data point using the trained MLP classifier.

        Parameters
        ----------
        grid_point : array-like
            1D sequence of feature values for prediction. Ignores last element as it represents a label.

        Returns
        -------
        float
            Predicted class label.
        """

        # Extract features (excluding the label if present)
        features = np.array(grid_point[:len(grid_point) - 1])

        # Reshape to 2D array with shape (1, n_features)
        features = features.reshape(1, -1)

        # Predict using the trained model
        prediction = self.best_mlp.predict(features)

        return float(prediction[0])  # Return the predicted label
