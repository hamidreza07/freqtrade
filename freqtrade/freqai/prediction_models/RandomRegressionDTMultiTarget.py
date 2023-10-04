import logging
import random
from typing import Any, Dict
import itertools

from sklearn.tree import DecisionTreeRegressor
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from math import sqrt
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class RandomRegressionDTMultiTarget(BaseRegressionModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        x_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        best_rmse_train = float('inf')
        best_rmse_test = float('inf')
        best_params = {}
        best_model = None  # Initialize the best model

        max_depths = [None] + [i for i in range(10, 100, 2)]
        min_samples_splits = [i for i in range(2, 20, 2)]
        min_samples_leafs = [i for i in range(2, 20, 2)]
        max_leaf_nodes = [i for i in range(2, 20, 2)]


        # Generate all possible combinations of parameters
        parameter_combinations = list(itertools.product(max_depths, min_samples_splits, min_samples_leafs,max_leaf_nodes))

        # Number of random parameter combinations to try
        num_random_combinations = 100

        # Randomly sample parameter combinations
        random_combinations = random.sample(parameter_combinations, num_random_combinations)

        for max_depth, min_samples_split, min_samples_leaf,max_leaf_nodes in random_combinations:
            # Initialize DecisionTreeRegressor with random parameters
            dt_regressor = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes = max_leaf_nodes
            )

            model = FreqaiMultiOutputRegressor(estimator=dt_regressor)
            fit_params = [{} for _ in range(y_train.shape[1])]

            # Fit the model with the current random parameter combination
            model.fit(X=x_train, y=y_train, sample_weight=sample_weight, fit_params=fit_params)

            # Calculate RMSE for training dataset
            y_train_pred = model.predict(x_train)
            train_rmse_predict = sqrt(mean_squared_error(y_train, y_train_pred))

            y_test = model.predict(data_dictionary["test_features"])
            test_rmse_predict = sqrt(mean_squared_error(y_test, data_dictionary["test_labels"]))

            if train_rmse_predict < best_rmse_train:
                best_rmse_train = train_rmse_predict
                best_rmse_test = test_rmse_predict

                best_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_leaf_nodes":max_leaf_nodes
                }
                best_model = model  # Update the best model

        logger.info(f"Best rmse train: {best_rmse_train}")
        logger.info(f"Best rmse test: {best_rmse_test}")
        logger.info(f"Best Parameters: {best_params}")

        return best_model
