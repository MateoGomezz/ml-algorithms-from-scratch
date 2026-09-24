# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        coeficiente = (np.cov(X,y)[0][1] / np.var(X, ddof=1))
        print(coeficiente)
        intercepto = np.mean(y) - coeficiente*np.mean(X)
        self.coefficients = coeficiente
        self.intercept = intercepto

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        X = np.column_stack((np.ones(len(X)),X)) #metemos columnas de unos
        estimaciones_params = ( np.linalg.inv(np.transpose(X)@X) ) @ np.transpose(X) @ y
        
        
       
        self.intercept = estimaciones_params[0]
        self.coefficients = estimaciones_params[1:]
        
        

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
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
    rss = np.sum(np.square((y_true-y_pred)))
    tss = np.sum(np.square((y_true-np.mean(y_true))))
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    mse = np.sum(np.square((y_true-y_pred)))/len(y_true)
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.sum(abs(y_pred-y_true))/len(y_true)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    ### Compare your model with sklearn linear regression model
    from sklearn.linear_model import LinearRegression
    
    # Assuming your data is stored in x and y
    x_reshaped = np.array([[dato] for dato in x])
    

    # Create and train the scikit-learn model
    sklearn_model = LinearRegression()
    
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    datasets = ["I","II","III","IV"]
    
    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        
        data = anscombe[anscombe["dataset"]==dataset]
        
        # Create a linear regression model
        model = LinearRegressor()

        # Fit the model
        X = data["x"]  # Predictor, make it 1D for your custom model
        y = data["y"]  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    #return results
    return anscombe,datasets,models,results


# Go to the notebook to visualize the results
