import logging
import sys
from pathlib import Path
from typing import Any, Dict

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np

logger = logging.getLogger(__name__)


class GridCatboostClassifierMultiTarget(BaseClassifierModel):
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
   
        iterations = [i for i in range(10, 100, 30)]
        depths = [i for i in range(2, 16, 8)]
        learning_rates = [0.001,0.005, 0.01]

        colsample_bylevels = [0.6, 0.8]

        best_accuracy_train = 0
        best_accuracy_test = 0
        best_params = {}
        best_model = None  # Initialize the best model

        for iteration in iterations:
            for depth in depths:
                for learning_rate in learning_rates:
                        for colsample_bylevel in colsample_bylevels:


                            cbc = CatBoostClassifier(
                                allow_writing_files=True,
                                loss_function='MultiClass',
                                train_dir=Path(dk.data_path),
                                iterations=iteration,
                                depth=depth,
                                learning_rate=learning_rate,
                                colsample_bylevel=colsample_bylevel
                                ,verbose=0)

                            X = data_dictionary["train_features"]
                            y = data_dictionary["train_labels"]

                            sample_weight = data_dictionary["train_weights"]

                            eval_sets = [None] * y.shape[1]

                            if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
                                eval_sets = [None] * data_dictionary['test_labels'].shape[1]

                                for i in range(data_dictionary['test_labels'].shape[1]):
                                    eval_sets[i] = Pool(
                                        data=data_dictionary["test_features"],
                                        label=data_dictionary["test_labels"].iloc[:, i],
                                        weight=data_dictionary["test_weights"],
                                    )

                            init_model = self.get_init_model(dk.pair)

                            if init_model:
                                init_models = init_model.estimators_
                            else:
                                init_models = [None] * y.shape[1]

                            fit_params = []
                            for i in range(len(eval_sets)):
                                fit_params.append({
                                    'eval_set': eval_sets[i], 'init_model': init_models[i],
                                    'log_cout': sys.stdout, 'log_cerr': sys.stderr,
                                })

                            model = FreqaiMultiOutputClassifier(estimator=cbc)
                            thread_training = self.freqai_info.get('multitarget_parallel_training', False)
                            if thread_training:
                                model.n_jobs = y.shape[1]
                            model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)
                            # Calculate accuracy for training dataset
                            y_train_pred = model.predict(X)
                            train_accuracy_predict = self.accuracy_score(y, y_train_pred)
                            if not all(item is None for item in eval_sets):

                                y_test = model.predict(data_dictionary["test_features"])

                                test_accuracy_predict = self.accuracy_score(y_test, data_dictionary["test_labels"])


                            if train_accuracy_predict > best_accuracy_test :
                                best_accuracy_train = train_accuracy_predict
                                if not all(item is None for item in eval_sets):
                                    best_accuracy_test = test_accuracy_predict
                                best_params = {
                                    'iterations': iteration,
                                    'depth': depth,
                                    'learning_rate': learning_rate,
                                    'colsample_bylevel': colsample_bylevel,
                                }

                                best_model = model  # Update the best model

            logger.info(f"Best accuracy train: {best_accuracy_train:.2f}")
            if not all(item is None for item in eval_sets):
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
