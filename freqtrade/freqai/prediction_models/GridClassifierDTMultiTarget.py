import logging
from typing import Any, Dict

from sklearn.tree import DecisionTreeClassifier
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np

logger = logging.getLogger(__name__)

class GridClassifierDTMultiTarget(BaseClassifierModel):
    """
    User created prediction model. The class inherits IFreqaiModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """


        best_accuracy_train = 0
        best_accuracy_test = 0
        best_params = {}
        best_model = None  # Initialize the best model

        x_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]
        max_depths = [None] + [i for i in range(10, 100, 30)]
        min_samples_splits = [i for i in range(2, 20, 6)]
        min_samples_leafs = [i for i in range(2, 20, 6)]
        max_leaf_nodes = [i for i in range(2, 20, 6)]

        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    for max_leaf_node in max_leaf_nodes:
                        
                        # Initialize DecisionTreeRegressor with different parameters
                        dt_regressor = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_leaf_nodes = max_leaf_node
                        )

        

                        model = FreqaiMultiOutputClassifier(estimator=dt_regressor)
                        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
                        if thread_training:
                                model.n_jobs = y_train.shape[1]
                        fit_params = [{} for _ in range(y_train.shape[1])]

                        # Fit the model with the current parameter combination
                        model.fit(X=x_train, y=y_train, sample_weight=sample_weight, fit_params=fit_params)

                        # Calculate accuracy for training dataset
                        y_train_pred = model.predict(x_train)
                        train_accuracy_predict = self.accuracy_score(y_train, y_train_pred)
                        if not data_dictionary['test_features'].empty:
                            y_test = model.predict(data_dictionary["test_features"])

                            test_accuracy_predict = self.accuracy_score(y_test, data_dictionary["test_labels"])


                        if train_accuracy_predict > best_accuracy_train :
                            best_accuracy_train = train_accuracy_predict
                            if not data_dictionary['test_features'].empty:
                                best_accuracy_test = test_accuracy_predict

                            best_params = {
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "min_samples_leaf":min_samples_leaf
                            }
                            best_model = model  # Update the best model

        logger.info(f"Best accuracy train: {best_accuracy_train:.2f}")
        if not data_dictionary['test_features'].empty:
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