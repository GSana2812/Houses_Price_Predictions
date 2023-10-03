#model_evaluation/model_evaluator
import pandas as pd
import matplotlib.pyplot as plt
from model.linear_regression import  LinearRegression
import seaborn as sns

class ModelEvaluator:

    @staticmethod
    def scatter_plot(y_true: pd.Series, y_pred: pd.Series):

        """
        Building a scatter plot with the results
        """

        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(y_true, y_true, color='black', linewidth=2, label='Regression Line')
        plt.xlabel('y_test', size=12)
        plt.ylabel('y_pred', size=12)
        plt.title('Predicted Values vs Original Values (Test Set)', size=15)
        plt.show()

    @staticmethod
    def weights_plot(model: LinearRegression, X_test: pd.DataFrame):

        """
        Building a bar plot with changes of coefficients indicating the impact they will have at the price
        """

        weights = model.weights
        feature_names = X_test.columns
        coefficients = pd.Series(weights, feature_names).sort_values()
        plt.figure(figsize=(10, 6))
        coefficients.plot(kind="bar", title="Model Coefficiants")
        plt.title('Model Coefficients')
        plt.show()




