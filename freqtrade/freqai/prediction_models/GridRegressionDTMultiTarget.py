import logging
from typing import Any, Dict

from sklearn.tree import DecisionTreeRegressor
import math
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from math import sqrt
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class GridRegressionDTMultiTarget(BaseRegressionModel):
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


        best_rmse_train = float('inf')
        best_rmse_test = float('inf')
        best_params = {}
        best_model = None  # Initialize the best model

        x_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]
        max_depths = [None] + [i for i in range(10, 100, 30)]
        min_samples_splits = [i for i in range(2, 20, 7)]
        min_samples_leafs = [i for i in range(2, 20, 7)]
        max_leaf_nodes = [i for i in range(2, 20, 7)]
        logger.info(f"total models: {len(max_depths)*len(min_samples_leafs)*len(min_samples_splits)*len(max_leaf_nodes)}")
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    for max_leaf_node in max_leaf_nodes:
                        
                        # Initialize DecisionTreeRegressor with different parameters
                        dt_regressor = DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_leaf_nodes = max_leaf_node
                        )

        

                        model = FreqaiMultiOutputRegressor(estimator=dt_regressor)
                        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
                        if thread_training:
                                model.n_jobs = y_train.shape[1]
                        fit_params = [{} for _ in range(y_train.shape[1])]

                        # Fit the model with the current parameter combination
                        model.fit(X=x_train, y=y_train, sample_weight=sample_weight, fit_params=fit_params)

                        # Calculate RMSE for training dataset
                        y_train_pred = model.predict(x_train)
                        train_rmse_predict = sqrt(mean_squared_error(y_train, y_train_pred))
                        if not data_dictionary['test_features'].empty:

                            y_test = model.predict(data_dictionary["test_features"])

                            test_rmse_predict = sqrt(mean_squared_error(y_test, data_dictionary["test_labels"]))


                        if train_rmse_predict < best_rmse_train :
                            best_rmse_train = train_rmse_predict
                            if not data_dictionary['test_features'].empty:
                                best_rmse_test = test_rmse_predict

                            best_params = {
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "min_samples_leaf":min_samples_leaf
                            }
                            best_model = model  # Update the best model

        logger.info(f"Best rmse train: {best_rmse_train}")
        if not data_dictionary['test_features'].empty:
            logger.info(f"Best rmse test: {best_rmse_test}")

        logger.info(f"Best Parameters: {best_params}")

        return best_model