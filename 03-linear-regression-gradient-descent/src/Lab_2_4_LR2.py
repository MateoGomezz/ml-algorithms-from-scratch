import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
        # X = np.column_stack((np.ones(len(X)),X)) #metemos columnas de unos
        estimaciones_params = ( np.linalg.inv(np.transpose(X)@X) ) @ np.transpose(X) @ y
        
        # Store the intercept and the coefficients of the model
        self.intercept = estimaciones_params[0]
        self.coefficients = estimaciones_params[1:]
        
    """
    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Gradient descent
        for epoch in range(iterations):
            predictions = self.predict(X)
            error = predictions - y

            
            
            grad_intercept = (2 / m) * np.sum(error)              
            grad_coef = (2 / m) * (X.T @ error) 
            
            self.intercept -= learning_rate* grad_intercept
            self.coefficients -= learning_rate*grad_coef

            if epoch % 1000 == 0:

                mse = (np.sum((y-predictions)**2))/len(X)
                print(f"Epoch {epoch}: MSE = {mse}")
    """
                
    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        X_features = X[:, 1:]

        m = len(y)
        self.coefficients = (np.random.rand(X.shape[1] - 1) * 0.01)  
        self.intercept = np.random.rand() * 0.01

        for epoch in range(iterations):
            predictions = self.predict(X_features)
            error = predictions - y

            grad_intercept = (2 / m) * np.sum(error)
            grad_coef = (2 / m) * (X_features.T @ error)

            
            self.intercept -= learning_rate * grad_intercept
            self.coefficients -= learning_rate * grad_coef

            if epoch % 1000 == 0:
                mse = np.sum((y - predictions) ** 2) / m
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        
        if np.ndim(X) == 1:
            y = self.intercept + self.coefficients*X
            predictions = y
        else:
            y = self.intercept + X@self.coefficients
            predictions = y
        return predictions



def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-np.mean(y_true))**2)
    r_squared = 1 - (rss/tss)
    
    # Root Mean Squared Error
    mse = np.sum((y_true-y_pred)**2)/len(y_true)
    rmse = np.sqrt(mse)


    # Mean Absolute Error
    mae = np.sum(abs(y_pred-y_true))/len(y_true)


    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    
    for index in sorted(categorical_indices, reverse=True):
        categorical_column = [X_transformed[i][index] for i in range(len(X_transformed))]
  
        unique_values = np.unique(categorical_column)

        one_hot = np.zeros((len(X_transformed),len(unique_values)))
        for i,categoria in enumerate(categorical_column):
            for j,cat_one_hot in enumerate(unique_values):
                if cat_one_hot == categoria:
                    one_hot[i][j] = 1
      
        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]
        
        X_sin_columna = np.delete(X_transformed,index,axis=1)
        
        X_transformed = []
        for fila,one_hot_fila in zip(X_sin_columna,one_hot):
            fila_izquierda = fila[:index]
            fila_derecha = fila[index:]
            X_transformed.append(np.concatenate((fila_izquierda,one_hot_fila,fila_derecha)))
        X_transformed = np.array(X_transformed)

    return X_transformed
