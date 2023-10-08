import logging
from typing import Any, Dict
import itertools
from sklearn.tree import DecisionTreeClassifier
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import random
logger = logging.getLogger(__name__)


class RandomClassifierDTMultiTarget(BaseClassifierModel):

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:

        max_depths = [None] + [i for i in range(10, 100, 10)]
        min_samples_splits = [i for i in range(2, 20, 3)]
        min_samples_leafs = [i for i in range(2, 20, 3)]
        max_leaf_nodes = [i for i in range(2, 20, 3)]

        # Generate all possible combinations of parameters using itertools.product
        parameter_combinations = list(itertools.product(max_depths, min_samples_splits, min_samples_leafs, max_leaf_nodes))
        selected_parameter_combinations = random.sample(parameter_combinations, 200)
        best_accuracy_train = float('inf')
        best_accuracy_test = float('inf')
        best_params = {}
        best_model = None  # Initialize the best model

        x_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        for max_depth, min_samples_split, min_samples_leaf, max_leaf_node in selected_parameter_combinations:
 
            # Initialize DecisionTreeRegressor with different parameters
            dt_regressor = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_node
            )

            model = FreqaiMultiOutputClassifier(estimator=dt_regressor)
            fit_params = [{} for _ in range(y_train.shape[1])]

            # Fit the model with the current parameter combination
            model.fit(X=x_train, y=y_train, sample_weight=sample_weight, fit_params=fit_params)

            # Calculate accuracy for training dataset
            y_train_pred = model.predict(x_train)
            train_accuracy_predict = self.accuracy_score(y_train, y_train_pred)

            y_test = model.predict(data_dictionary["test_features"])

            test_accuracy_predict = self.accuracy_score(y_test, data_dictionary["test_labels"])

            if train_accuracy_predict < best_accuracy_train:
                best_accuracy_train = train_accuracy_predict
                best_accuracy_test = test_accuracy_predict

                best_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_leaf_nodes": max_leaf_node
                }
                best_model = model  # Update the best model

        logger.info(f"Best accuracy train: {best_accuracy_train:.2f}")
        logger.info(f"Best accuracy test: {best_accuracy_test:.2f}")
        logger.info(f"Best Parameters: {best_params}")

        return best_model
    def accuracy_score(self, y_true, y_pred):
        """
        Calculate the accuracy score for multiclass-multioutput classification.
        :param y_true: Ground truth labels (array-like of shape (n_samples, n_outputs))
        :param y_pred: Predicted labels (array-like of shape (n_samples, n_outputs))
        :return: Accuracy score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check if dimensions match
        if y_true.shape != y_pred.shape:
            raise ValueError("Input shapes do not match.")

        # Calculate accuracy for each output and then average
        accuracies = []
        for i in range(y_true.shape[1]):
            correct = np.sum(y_true[:, i] == y_pred[:, i])
            total = len(y_true)
            accuracy = correct / total
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        return mean_accuracy