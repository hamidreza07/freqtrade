import logging
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class OPTMLPRegressor(BaseRegressionModel):
    """
    User created prediction model. The class inherits IFreqaiModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """


        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].iloc[:, -1]

        # Define the hyperparameter grid to search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'invscaling'],
            'max_iter': [200, 300],
            # Add other hyperparameters and their values as needed
        }
        model = MLPRegressor()

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2)

        grid_search.fit(X, y)

        # Get the best hyperparameters and best model from grid search
        best_params = grid_search.best_params_
        logger.info(f"best params: {best_params}")
        best_model = grid_search.best_estimator_

        best_model.fit(X, y)

        return best_model
