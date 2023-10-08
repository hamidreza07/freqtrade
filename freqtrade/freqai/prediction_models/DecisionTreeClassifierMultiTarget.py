import logging
from typing import Any, Dict

from sklearn.tree import DecisionTreeClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class DecisionTreeClassifierMultiTarget(BaseClassifierModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        dt_classifier = DecisionTreeClassifier(**self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        fit_params = [{} for _ in range(y.shape[1])]
        model = FreqaiMultiOutputClassifier(estimator=dt_classifier)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model
