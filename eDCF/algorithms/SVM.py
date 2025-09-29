import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


class SVM:
    """
    Provide functionality to train, evaluate, and predict using a Support Vector Machine (SVM) classifier
    with hyperparameter tuning via scikit-learn's SVC and GridSearchCV.

    Parameters
    ----------
    C : list[float]
        Candidate regularization parameter values.
    kernel : list[str]
        Kernel types to consider (e.g., ['linear', 'rbf']).
    gamma : list[float]
        Kernel coefficient values to search.
    decision_function_shape : list[str]
        Decision function shapes ('ovo', 'ovr') to consider.

    Attributes
    ----------
    best_svm : SVC
        Fitted SVM classifier with optimal hyperparameters.
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
        Perform hyperparameter tuning via GridSearchCV and fit the SVM.
    train() -> None
        Retrain the SVM on the full training data using best hyperparameters.
    initialize_hyper() -> None
        Load saved hyperparameters and retrain the SVM classifier.
    execute(grid_point: array-like) -> float
        Predict the class label for a single data point.

    Examples
    --------
    >>> svm = SVM([0.1, 1, 10], ['linear', 'rbf'], [0.001, 0.01], ['ovr'])
    >>> svm.train_hyper()
    >>> label = svm.execute([x1, x2, x3])
    """

    def __init__(self, C, kernel, gamma, decision_function_shape):
        """
        Initialize SVM with specified hyperparameter search spaces.

        Parameters
        ----------
        C : list[float]
            Candidate regularization parameter values.
        kernel : list[str]
            Kernel types to consider (e.g., 'linear', 'rbf').
        gamma : list[float]
            Kernel coefficient values to search.
        decision_function_shape : list[str]
            Decision function shapes ('ovo', 'ovr') to evaluate.
        """
        
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape

        # Placeholders for best hyperparameters and trained models
        self.best_params = ...
        self.best_svm = ...
        self.X_test = ...
        self.X_train = ...
        self.y_test = ...
        self.y_train = ...

    def train_hyper(self, core_ctrl: int = -1) -> None:
        """
        Tune and evaluate the SVM classifier using GridSearchCV.

        Workflow:
        1. Load 'Datapoints.npy' and stack into array.
        2. Split into features (X) and labels (y).
        3. Perform 80/20 train/test split.
        4. Define parameter grid for C, kernel, gamma, decision_function_shape.
        5. Execute GridSearchCV with `self.cvn` folds and `core_ctrl` parallel jobs.
        6. Store the best estimator in `self.best_svm` and parameters in `self.best_params`.
        7. Save test score, hyperparameters, and train/test splits to files.

        Parameters
        ----------
        core_ctrl : int, optional
            Number of parallel jobs for GridSearchCV (default: -1 uses all CPU cores).

        Returns
        -------
        None
        """

        # Load data from 'Datapoints.npy'
        data = np.load('Datapoints.npy', allow_pickle=True)
        data = np.vstack(data)  # Combine data into a single array

        # Split data into features (X) and labels (y)
        X = data[:, :-1]  # All columns except the last are features
        y = data[:, -1].astype(int).reshape(len(data))  # The last column is the label

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the parameter grid for tuning
        parameter_space_poly = {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'decision_function_shape': self.decision_function_shape
        }

        # Initialize the SVM classifier
        svm = SVC()

        # Define k-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Set parameter space and kernel type for logging
        parameter_space = parameter_space_poly
        kernel_type = 'Radial Basis Function Kernel'

        # Perform GridSearchCV for hyperparameter tuning
        clf = GridSearchCV(svm, parameter_space, cv=cv, scoring='f1_macro', n_jobs=core_ctrl)

        # Train the model with GridSearchCV
        clf.fit(self.X_train, self.y_train)

        # Output the best parameters and cross-validation scores
        print('Best parameters using', kernel_type, 'found:\n', clf.best_params_)
        print("Best cross-validation score using", kernel_type, "found:", clf.best_score_)

        # Evaluate the best model on the test set
        test_score = clf.score(self.X_test, self.y_test)
        print("Test accuracy before retraining (scikit-learn):", test_score)

        # Save test score and best hyperparameters
        f1_score = np.asarray(test_score)  # Convert score to array for saving
        np.save("algorithms/Testing_Score", f1_score)
        np.save("algorithms/Hyper_Param", clf.best_params_)

        # Save train-test split data
        np.save('algorithms/X_train', self.X_train)
        np.save('algorithms/y_train', self.y_train)
        np.save('algorithms/X_test', self.X_test)
        np.save('algorithms/y_test', self.y_test)

        return None

    def train(self) -> None:
        """
        Retrain the SVM classifier on the full training set using best hyperparameters.

        Workflow:
        1. Load training data from 'algorithms/X_train.npy' and 'algorithms/y_train.npy'.
        2. Initialize SVC with `self.best_params`.
        3. Fit the classifier on the entire training set.

        Returns
        -------
        None
        """

        # Load previously saved training data
        self.X_train = np.load('algorithms/X_train.npy', allow_pickle=True)
        self.y_train = np.load('algorithms/y_train.npy', allow_pickle=True)

        # Initialize the SVM with the best hyperparameters
        self.best_svm = SVC(**self.best_params)

        # Train the model on the entire training set
        self.best_svm.fit(self.X_train, self.y_train)

        return None

    def initialize_hyper(self) -> None:
        """
        Load saved hyperparameters and retrain the SVM classifier.

        Workflow:
        1. Load hyperparameters dict from 'algorithms/Hyper_Param.npy' into `self.best_params`.
        2. Call `self.train()` to fit the classifier on the full training set.

        Returns
        -------
        None
        """

        # Load the best hyperparameters from file
        self.best_params = np.load("algorithms/Hyper_Param.npy", allow_pickle=True)
        self.best_params = self.best_params.item()  # Convert numpy object to dictionary

        # Retrain the model with the loaded hyperparameters
        self.train()

        return None

    def execute(self, grid_point) -> float:
        """
        Predict the class label for a single data point using the trained SVM classifier.

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

        # Reshape features to 2D array for prediction
        features = features.reshape(1, -1)

        # Predict the class label using the trained SVM
        prediction = self.best_svm.predict(features)

        return float(prediction[0])  # Return the predicted label
