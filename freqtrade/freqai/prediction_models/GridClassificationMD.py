import logging
from typing import Any, Dict
from sklearn.neural_network import MLPClassifier  # Change to MLPClassifier for classification
from sklearn.tree import DecisionTreeClassifier  # Change to DecisionTreeClassifier for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score  # Change to classification metric
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel  # Import classification base model
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype

logger = logging.getLogger(__name__)

class GridClassificationMD(BaseClassifierModel):  # Change to a classification base model
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

        mlp_param_grid = {
            'hidden_layer_sizes': [(50, 100), (100, 200)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'max_iter': [500],  # Increase max_iter values
        }

        decision_tree_param_grid = {
            'max_depth': [ 5,  15],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1,  4],
        }

        # Train MLPClassifier for classification
        mlp_model = self.train_model(MLPClassifier(), mlp_param_grid, X, y, 'MLPClassifier')

        # Train DecisionTreeClassifier for classification
        decision_tree_model = self.train_model(DecisionTreeClassifier(), decision_tree_param_grid, X, y, 'DecisionTreeClassifier')

        # Train RandomForestClassifier for classification

        # Compare accuracy and select the best model
        best_model = self.select_best_model([mlp_model, decision_tree_model], X, y)

        logger.info(f"Chosen model: {best_model.__class__.__name__}")

        return best_model

    def train_model(self, model, param_grid, X, y, model_name):
        """
        Train a classification model using GridSearchCV and return the best model.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=2)
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
