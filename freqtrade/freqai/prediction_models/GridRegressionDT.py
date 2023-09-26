import logging
from typing import Any, Dict
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
logger = logging.getLogger(__name__)
# Define a custom scoring function for RMSE
def rmse_scorer(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return -rmse  # Negate the RMSE to use as a scoring metric in GridSearchCV

class GridRegressionDT(BaseRegressionModel):
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

        # Split your test data into test features and labels
        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"].iloc[:, -1]

        decision_tree_param_grid = {
            'max_depth': [None] +[i for i in range(10,100,5)],  # Adjust the range as needed
            'min_samples_split': [i for i in range(2,20,2)],
            'min_samples_leaf': [i for i in range(2,20,2)]
        }

        # Train DecisionTreeRegressor
        decision_tree_model = self.train_model(DecisionTreeRegressor(), decision_tree_param_grid, X, y, X_test, y_test, 'DecisionTreeRegressor')

        return decision_tree_model

    def train_model(self, model, param_grid, X, y, X_test, y_test, model_name):
        """
        Train a regression model using GridSearchCV and return the best model.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(rmse_scorer, greater_is_better=False), cv=2, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        logger.info(f"Best Score: {grid_search.best_score_}")

        best_model = grid_search.best_estimator_
        best_model.fit(X, y)

        # Evaluate the best model on the test data
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Test RMSE for {model_name}: {test_rmse}")

        return best_model
