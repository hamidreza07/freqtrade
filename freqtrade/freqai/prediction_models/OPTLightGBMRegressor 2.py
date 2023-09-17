import logging
from typing import Any, Dict
from lightgbm import LGBMRegressor
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class OPTLightGBMRegressor2(BaseRegressionModel):
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

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = data_dictionary["test_weights"]
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        init_model = self.get_init_model(dk.pair)

        # Define the hyperparameter grid to search
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'min_child_samples': [1, 5, 10, 20],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.2, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.2, 0.5, 1.0],
            'num_leaves': [31, 50, 75, 100],
            # Add other hyperparameters and their values as needed
        }
        model = LGBMRegressor(**self.model_training_parameters)

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=2)

        grid_search.fit(X, y, sample_weight=train_weights)

        # Get the best hyperparameters and best model from grid search
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        best_model.fit(X, y, eval_set=eval_set, sample_weight=train_weights, eval_sample_weight=[eval_weights])

        return best_model
