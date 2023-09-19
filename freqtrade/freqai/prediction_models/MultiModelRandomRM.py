import logging
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Use RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint  # Import a random integer distribution
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
logger = logging.getLogger(__name__)

class RandomSelectRegressor(BaseRegressionModel):
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
        mlp_param_dist = {
            'hidden_layer_sizes': sp_randint(50, 200),  # Random integer between 50 and 200
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'invscaling'],
            'max_iter': sp_randint(200, 400),  # Random integer between 200 and 400
            # Add other hyperparameters and their distributions as needed
        }

        random_forest_param_dist = {
            'n_estimators': sp_randint(50, 200),  # Random integer between 50 and 200
            'max_depth': [None] + list(range(5, 16)),  # Include None and integers from 5 to 15
            'min_samples_split': sp_randint(2, 11),  # Random integer between 2 and 10
            'min_samples_leaf': sp_randint(1, 5),  # Random integer between 1 and 4
            # Add other hyperparameters and their distributions as needed
        }

        # Train MLPRegressor with RandomizedSearchCV
        mlp_model = self.train_model(MLPRegressor(), mlp_param_dist, X, y, 'MLPRegressor')


        # Train RandomForestRegressor with RandomizedSearchCV
        random_forest_model = self.train_model(RandomForestRegressor(), random_forest_param_dist, X, y, 'RandomForestRegressor')

        # Compare RMSE and select the best model
        models = {
            'MLPRegressor': mlp_model,
            'RandomForestRegressor': random_forest_model
        }
        best_model_name = min(models, key=lambda model_name: self.calculate_rmse(models[model_name], X, y))

        logger.info(f"Chosen model: {best_model_name}")

        return models[best_model_name]

    def train_model(self, model, param_dist, X, y, model_name):
        """
        Train a regression model using RandomizedSearchCV and return the best model.
        """
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,  # Adjust the number of iterations as needed
            scoring='neg_mean_squared_error',
            cv=2,
            random_state=42,
              n_jobs=-1 
        )
        randomized_search.fit(X, y)

        best_params = randomized_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        best_model = randomized_search.best_estimator_
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
