import logging
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np

logger = logging.getLogger(__name__)

# Define a custom scoring function for RMSE
def rmse_scorer(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return -rmse  # Negate the RMSE to use as a scoring metric in GridSearchCV

class RandomRegressionMLP(BaseRegressionModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        # Split data into training and testing sets
        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"].iloc[:, -1]
        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"].iloc[:, -1]

        # Define the hyperparameter grids for different models
        mlp_param_grid = {
            'hidden_layer_sizes': [(50, 100), (300, 400)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [f for f in np.arange(0.0001, 0.001, 0.00029)],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [500],
        }

        # Train MLPRegressor
        mlp_model = self.train_model(MLPRegressor(), mlp_param_grid, X_train, y_train, X_test, y_test, 'MLPRegressor')

        return mlp_model

    def train_model(self, model, param_grid, X_train, y_train, X_test, y_test, model_name):
        # Define the GridSearchCV with custom RMSE scorer
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(rmse_scorer, greater_is_better=False), cv=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_  # Convert back to positive RMSE
        logger.info(f"Best params for {model_name}: {best_params}")
        logger.info(f"Best RMSE Score: {best_score}")

        best_model = grid_search.best_estimator_

        # Evaluate the best model on the test set
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"RMSE on test set: {rmse}")

        return best_model
