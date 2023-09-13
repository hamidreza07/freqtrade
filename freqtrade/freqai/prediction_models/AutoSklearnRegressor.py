import logging
from typing import Any, Dict
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import *

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class AutoSklearnCustomRegressor(BaseRegressionModel):
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
            X_test = y_test = None
        else:
            X_test,y_test = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]


               
        model = AutoSklearnRegressor(**self.model_training_parameters,metric=mean_squared_error)


        model.fit(X=X, y=y,X_test=X_test,y_test=y_test)

        return model
