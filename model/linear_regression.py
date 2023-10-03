import numpy as np
import pandas as pd

class LinearRegression:

    """
    Linear Regression implementation using gradient descent.

    Parameters
    ---------
    - lr (float): Learning rate for gradient descent.
    - n_iters (int): Number of iterations for gradient descent.

    Attributes
    ---------
    - lr (float): Learning rate.
    - n_iters (int): Number of iterations.
    - weights (numpy.ndarray): Coefficients for each feature.
    - bias (float): Intercept term.
    """

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):

        """
        Initialize the LinearRegression object.

        Parameters
        ----------
        - lr (float): Learning rate for gradient descent.
        - n_iters (int): Number of iterations for gradient descent.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        """
        Fit the linear regression model to the training data.

        Parameters
        ----------
        - X (numpy.ndarray or pandas.DataFrame): Training feature matrix.
        - y (numpy.ndarray or pandas.Series): Target values.
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: pd.DataFrame) -> pd.Series:

        """
        Make predictions on new data.

        Parameters
        ----------
        - X (numpy.ndarray or pandas.DataFrame): Feature matrix for prediction.

        Returns
        ---------
        - numpy.ndarray: Predicted values.
        """

        return X.dot(self.weights) + self.bias

    @staticmethod
    def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:

        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Parameters
        ---------
        - y_true (numpy.ndarray or pandas.Series): True target values.
        - y_pred (numpy.ndarray or pandas.Series): Predicted target values.

        Returns
        --------
        - float: Mean Squared Error (MSE).
        """

        return np.mean((y_true - y_pred) ** 2)

    @staticmethod

    def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:

        """
        Calculate the Mean Absolute Error (MAE) between true and predicted values.

        Parameters
        ----------
        - y_true (numpy.ndarray or pandas.Series): True target values.
        - y_pred (numpy.ndarray or pandas.Series): Predicted target values.

        Returns
        --------
        - float: Mean Absolute Error (MAE).
        """

        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r_squared(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculate the R-squared (coefficient of determination) between true and predicted values.

        Parameters
        ----------
        - y_true (numpy.ndarray or pandas.Series): True target values.
        - y_pred (numpy.ndarray or pandas.Series): Predicted target values.

        Returns
        ---------
        - float: R-squared value.
        """

        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ssr / sst)