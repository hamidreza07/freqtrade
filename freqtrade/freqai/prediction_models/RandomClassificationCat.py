import logging
from typing import Any, Dict, Tuple
import numpy.typing as npt
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict
from sklearn.model_selection import RandomizedSearchCV
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from catboost import CatBoostClassifier  # Import CatBoostClassifier from catboost

logger = logging.getLogger(__name__)

class RandomClassificationCat(BaseClassifierModel):
    """
    Automatically selects the best classification model based on accuracy or F1 score.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired models here.
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        # le = LabelEncoder()
        # if not is_integer_dtype(y):
        #     y = pd.Series(le.fit_transform(y.iloc[:, -1]), dtype="int64")

        # Train CatBoost Classifier with GridSearchCV
        best_model = self.train_model(X, y)

        return best_model

    def train_model(self, X, y):
        """
        Train a classification model using GridSearchCV and return the best model.
        """
        catboost_param_grid = {
            'iterations': [50, 100],
            'depth': [3, 10],
            'learning_rate': [0.001, 0.01],
            'subsample': [0.9, 0.3],
            'colsample_bylevel': [0.6, 0.8],
            # Add other CatBoost-specific hyperparameters as needed
        }

        grid_search = RandomizedSearchCV(
            estimator=CatBoostClassifier(),
            param_distributions=catboost_param_grid,
            scoring='accuracy',  # You can use 'f1_micro', 'f1_macro', or other suitable scoring metric
            cv=2,
            n_jobs=-1
        )
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for CatBoost Classifier: {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)
        return best_model
