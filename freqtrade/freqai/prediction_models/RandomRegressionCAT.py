import logging
from typing import Any, Dict
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)


def rmse_scorer(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return -rmse  # Negate the RMSE to use as a scoring metric in GridSearchCV

class RandomRegressionCAT(BaseRegressionModel):
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
        y = data_dictionary["train_labels"]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            X_test = None
            y_test = None
        else:
            X_test, y_test = data_dictionary["test_features"], data_dictionary["test_labels"]



        catboost_param_grid = {
            'iterations': [i for i in range(50,110,50)],
            'depth': [i for i in range(3,14,5)],
            'learning_rate': [f for f in np.arange(0.0001, 0.001, 0.00029)],
            'subsample':[f for f in np.arange(0.6, 1, 0.2)],
            'colsample_bylevel': [f for f in np.arange(0.6, 1, 0.2)],
            # Add other CatBoost-specific hyperparameters as needed
        }



        catboost_model = self.train_model(CatBoostRegressor(), catboost_param_grid, X, y,X_test, y_test)


        return catboost_model

    def train_model(self, model, param_grid, X, y, X_test, y_test):
        """
        Train a regression model using GridSearchCV and return the best model.
        """
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=make_scorer(rmse_scorer, greater_is_better=False), cv=2, n_jobs=-1, verbose=1)

        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params : {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)

        # Check if X_test and y_test are not None or empty before evaluation
        if X_test is not None and y_test is not None and len(X_test) > 0 and len(y_test) > 0:
            # Evaluate the best model on the test data
            y_pred = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"Test RMSE: {test_rmse}")
        else:
            logger.info("Test data is empty or None, skipping evaluation.")

        return best_model