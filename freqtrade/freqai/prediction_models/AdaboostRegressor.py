import logging
import sys
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class AdaboostRegressor(BaseRegressionModel):
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

        train_data = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            test_data = None
            test_labels = None
        else:
            test_data = data_dictionary["test_features"]
            test_labels = data_dictionary["test_labels"]

        base_estimator = DecisionTreeRegressor(**self.model_training_parameters.get('base_estimator_params', {}))

        model = AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=self.model_training_parameters.get('n_estimators', 50),
            learning_rate=self.model_training_parameters.get('learning_rate', 1.0),
            random_state=self.model_training_parameters.get('random_state', None),
        )

        model.fit(X=train_data, y=train_labels)

        return model
