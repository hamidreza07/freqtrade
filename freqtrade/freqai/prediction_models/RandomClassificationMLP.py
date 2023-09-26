import logging
from typing import Any, Dict, Tuple
from sklearn.neural_network import MLPClassifier
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
from pandas import DataFrame
import numpy.typing as npt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class RandomClassificationMLP(BaseClassifierModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y.iloc[:, -1]), dtype="int64")

        mlp_param_grid = {
            'hidden_layer_sizes': [(50, 100),(300,400)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [f for f in np.arange(0.0001, 0.001, 0.00029)],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Added learning_rate options
            'max_iter': [500],  # Added max_iter options
        }

        mlp_model = self.train_model(MLPClassifier(), mlp_param_grid, X, y, 'MLPClassifier')

        best_model = self.select_best_model([mlp_model], X, y)

        return best_model

    def train_model(self, model, param_grid, X, y, model_name):
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', cv=2, n_jobs=-1)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)
        return best_model

    def select_best_model(self, models, X, y):
        best_model = None
        best_accuracy = 0.0

        for model in models:
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        logger.info(f"Best accuracy : {best_accuracy}")

        return best_model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        le = LabelEncoder()
        label = dk.label_list[0]
        labels_before = list(dk.data['labels_std'].keys())
        labels_after = le.fit_transform(labels_before).tolist()
        pred_df[label] = le.inverse_transform(pred_df[label])
        pred_df = pred_df.rename(
            columns={labels_after[i]: labels_before[i] for i in range(len(labels_before))})

        return (pred_df, dk.do_predict)
