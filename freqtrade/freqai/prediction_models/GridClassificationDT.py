import logging
from typing import Any, Dict,Tuple
from sklearn.neural_network import MLPClassifier  # Change to MLPClassifier for classification
from sklearn.tree import DecisionTreeClassifier  # Change to DecisionTreeClassifier for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score  # Change to classification metric
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel  # Import classification base model
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import numpy.typing as npt 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
from pandas import DataFrame
logger = logging.getLogger(__name__)

class GridClassificationDT(BaseClassifierModel):  # Change to a classification base model
    """
    Automatically selects the best classification model based on accuracy.
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

        decision_tree_param_grid = {
            'max_depth': [None] +[i for i in range(10,100,40)],  # Adjust the range as needed
            'min_samples_split': [i for i in range(2,20,5)],
            'min_samples_leaf': [i for i in range(2,20,5)]
        }
        # Train MLPClassifier for classification

        # Train DecisionTreeClassifier for classification
        decision_tree_model = self.train_model(DecisionTreeClassifier(), decision_tree_param_grid, X, y, 'DecisionTreeClassifier')

        # Train RandomForestClassifier for classification

        # Compare accuracy and select the best model
        best_model = self.select_best_model([decision_tree_model], X, y)

        return best_model

    def train_model(self, model, param_grid, X, y, model_name):
        """
        Train a classification model using GridSearchCV and return the best model.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        logger.info(f"Best params for {model_name}: {best_params}")
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)
        return best_model

    def select_best_model(self, models, X, y):
        """
        Select the best classification model based on accuracy.
        """
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