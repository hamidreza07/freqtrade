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
from sklearn.model_selection import  RandomizedSearchCV
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import xgboost as xgb

logger = logging.getLogger(__name__)

class RandomClassificationXGB(BaseClassifierModel):
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
        
        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y.iloc[:, -1]), dtype="int64")

        # Define the hyperparameter grids for different classification models
        xgb_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 10],
            'learning_rate': [0.001, 0.01],
            'subsample': [0.9, 0.3],
            'colsample_bytree': [0.6, 0.8],
            # Add other XGBoost-specific hyperparameters as needed
        }

        # Train XGBoost Classifier with GridSearchCV
        best_model = self.train_model(xgb.XGBClassifier(), xgb_param_grid, X, y, 'XGBoost Classifier')

        return best_model
 

    def train_model(self, model, param_grid, X, y, model_name):
        """
        Train a classification model using GridSearchCV and return the best model.
        """
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring='accuracy',  # You can use 'f1_micro', 'f1_macro', or other suitable scoring metric
            cv=2,
            n_jobs=-1
        )
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)
        return best_model


    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        le = LabelEncoder()
        label = dk.label_list[0]
        labels_before = list(dk.data['labels_std'].keys())
        labels_after = le.fit_transform(labels_before).tolist()
        pred_df[label] = le.inverse_transform(pred_df[label])
        pred_df = pred_df.rename(
            columns={labels_after[i]: labels_before[i] for i in range(len(labels_before))})

        return (pred_df, dk.do_predict)