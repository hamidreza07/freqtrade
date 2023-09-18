import logging
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
logger = logging.getLogger(__name__)

class AutoSelectRegressor(BaseRegressionModel):
    """
    Automatically selects the best regression model based on RMSE.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired models here.
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].iloc[:, -1]

        # Define the hyperparameter grids for different models
        mlp_param_grid = {
            'hidden_layer_sizes': [(50,100), (100,200)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'invscaling'],
            'max_iter': [200, 300],
            # Add other hyperparameters and their values as needed
        }
        decision_tree_param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            # Add other hyperparameters and their values as needed
        }

        # Train MLPRegressor
        mlp_model = self.train_model(MLPRegressor(), mlp_param_grid, X, y, 'MLPRegressor')

        # Train DecisionTreeRegressor
        decision_tree_model = self.train_model(DecisionTreeRegressor(), decision_tree_param_grid, X, y, 'DecisionTreeRegressor')

        # Compare RMSE and select the best model
        best_model = mlp_model if self.calculate_rmse(mlp_model, X, y) < self.calculate_rmse(decision_tree_model, X, y) else decision_tree_model
        
        logger.info(f"Chosen model: {'MLPRegressor' if best_model == mlp_model else 'DecisionTreeRegressor'}")

        return best_model

    def train_model(self, model, param_grid, X, y, model_name):
        """
        Train a regression model using GridSearchCV and return the best model.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)
        return best_model

    def calculate_rmse(self, model, X, y):
        """
        Calculate RMSE for a given model and data.
        """
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return rmse
