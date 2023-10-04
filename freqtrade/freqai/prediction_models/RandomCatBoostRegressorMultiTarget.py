import logging
import random
from typing import Any, Dict
import itertools
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor,Pool
import sys
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class RandomCatBoostRegressorMultiTarget(BaseRegressionModel):
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        best_rmse_train = float('inf')
        best_params = {}
        best_model = None  # Initialize the best model

        # Define the hyperparameter ranges for random search
        param_ranges = {
            'iterations': [i for i in range(2, 20, 3)],
            'depth': [i for i in range(2, 15, 2)],
            'learning_rate': [f for f in np.arange(0.0001, 0.001, 0.0002)],
        }

        # Generate all possible combinations of parameters
        param_combinations = list(itertools.product(param_ranges['iterations'],
                                                    param_ranges['depth'],
                                                    param_ranges['learning_rate']))

        # Number of random parameter combinations to try
        num_random_combinations = 10

        # Randomly sample parameter combinations
        random_combinations = random.sample(param_combinations, num_random_combinations)

        for iterations, depth, learning_rate in random_combinations:
            catboost = CatBoostRegressor(iterations=iterations,
                                         depth=depth,
                                         learning_rate=learning_rate,
                                         thread_count=-1,
                                         verbose=0)

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
                        'eval_set': eval_sets[i],  'init_model': init_models[i],
                        'log_cout': sys.stdout, 'log_cerr': sys.stderr,
                    })


            model = FreqaiMultiOutputRegressor(estimator=catboost)
            thread_training = self.freqai_info.get('multitarget_parallel_training', False)
            if thread_training:
                model.n_jobs = y.shape[1]
            model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

            # Calculate RMSE for training dataset
            y_train_pred = model.predict(X)
            train_rmse_predict = sqrt(mean_squared_error(y, y_train_pred))

            if train_rmse_predict < best_rmse_train:
                best_rmse_train = train_rmse_predict

                best_params = {
                    'iterations': iterations,
                    'depth': depth,
                    'learning_rate': learning_rate,
                }
                best_model = model  # Update the best model

        logger.info(f"Best rmse train: {best_rmse_train}")
        logger.info(f"Best Parameters: {best_params}")

        return best_model
