import logging
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class MLPRegressorModel(BaseRegressionModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        model = MLPRegressor(**self.model_training_parameters)

        model.fit(X=X, y=y)

        return model
