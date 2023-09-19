import logging
from typing import Any, Dict
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
import pandas as pd
import logging
from typing import Any, Dict
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import accuracy_score
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

class RandomClassificationXC(BaseClassifierModel):
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
            'n_estimators': [50],
            'max_depth': [3],
            'learning_rate': [0.001],
            'subsample': [0.9],
            'colsample_bytree': [0.6],
            # Add other XGBoost-specific hyperparameters as needed
        }

        catboost_param_grid = {
            'iterations': [50],
            'depth': [3],
            'learning_rate': [0.001],
            'subsample': [0.6],
            'colsample_bylevel': [0.9],
            # Add other CatBoost-specific hyperparameters as needed
        }

        # Train XGBoost Classifier with GridSearchCV
        xgb_model = self.train_model(xgb.XGBClassifier(), xgb_param_grid, X, y, 'XGBoost Classifier')

        # Train CatBoost Classifier with GridSearchCV
        catboost_model = self.train_model(CatBoostClassifier(), catboost_param_grid, X, y, 'CatBoost Classifier')

        # Compare accuracy or F1 score and select the best model
        models = {
            'XGBoost Classifier': xgb_model,
            'CatBoost Classifier': catboost_model,
        }
        best_model_name = max(models, key=lambda model_name: self.calculate_accuracy_or_f1(models[model_name], X, y))

        logger.info(f"Chosen model: {best_model_name}")

        return models[best_model_name]

    def train_model(self, model, param_grid, X, y, model_name):
        """
        Train a classification model using GridSearchCV and return the best model.
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
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

    def calculate_accuracy_or_f1(self, model, X, y):
        """
        Calculate accuracy or F1 score for a given classification model and data.
        """
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy 