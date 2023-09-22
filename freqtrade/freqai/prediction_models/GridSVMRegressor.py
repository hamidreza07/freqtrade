import logging
from typing import Any, Dict
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class GridSVMRegressor(BaseRegressionModel):

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Perform a grid search for hyperparameter tuning.
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        :return: Best trained model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].iloc[:, -1]
        train_weights = data_dictionary["train_weights"]

        # Define the parameter grid for the SVR model
        param_grid = {
            'C': [0.1, 10.0],
            'kernel': ['linear', 'rbf'],
            'degree': [2, 4],
            'gamma': [0.1, 1.0],
        }

        # Create an SVR model
        model = SVR()

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y, sample_weight=train_weights)
        best_params = grid_search.best_params_
        logger.info(f"best params: {best_params}")
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_

        return best_model
