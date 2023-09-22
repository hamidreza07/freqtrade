import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy.typing as npt
logger = logging.getLogger(__name__)

class GridSVCClassifier(BaseClassifierModel):
    """
    User created prediction model using Support Vector Classifier (SVC).
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0]

        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y), dtype="int64")

        train_weights = data_dictionary["train_weights"]

        # Define the hyperparameter grid for the grid search
        param_grid = {
            'C': [0.1, 10],
            'kernel': ['linear', 'sigmoid'],
            'gamma':  [0.01, 1]
        }

        # Instantiate the SVC model
        model = SVC(probability=True)

        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2,verbose=1)

        # Fit the grid search to the data
        grid_search.fit(X, y, sample_weight=train_weights)

        # Get the best estimator (model with the best hyperparameters)
        best_param = grid_search.best_params_
        logger.info(f"best param: {best_param}")
        best_model = grid_search.best_estimator_
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
