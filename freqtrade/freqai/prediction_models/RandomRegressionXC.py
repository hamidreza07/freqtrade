import logging
from typing import Any, Dict
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor


logger = logging.getLogger(__name__)

class RandomRegressionXC(BaseRegressionModel):
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
        xgb_param_dist = {
            'n_estimators': sp_randint(50, 200),
            'max_depth': sp_randint(3, 10),
            'learning_rate': [0.001, 0.01],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            # Add other XGBoost-specific hyperparameters as needed
        }

        catboost_param_dist = {
            'iterations': sp_randint(50, 200),
            'depth': sp_randint(3, 10),
            'learning_rate': [0.001, 0.01],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bylevel': [0.6, 0.7, 0.8],
            # Add other CatBoost-specific hyperparameters as needed
        }



        # Train XGBoost Regressor with RandomizedSearchCV
        xgb_model = self.train_model(xgb.XGBRegressor(), xgb_param_dist, X, y, 'XGBoost Regressor')

        # Train CatBoost Regressor with RandomizedSearchCV
        catboost_model = self.train_model(CatBoostRegressor(), catboost_param_dist, X, y, 'CatBoost Regressor')



        # Compare RMSE and select the best model
        models = {
            'XGBoost Regressor': xgb_model,
            'CatBoost Regressor': catboost_model
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
            n_iter=10,
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
