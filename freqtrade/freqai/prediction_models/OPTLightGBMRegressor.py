import logging
from typing import Any, Dict
from lightgbm import LGBMRegressor
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)

class OPTLightGBMRegressor(BaseRegressionModel):
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

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = data_dictionary["test_weights"]
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                # Add other hyperparameters to tune
            }

            init_model = self.get_init_model(dk.pair)

            model = LGBMRegressor(**params)

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], sample_weight=train_weights,
                      eval_sample_weight=[eval_weights], init_model=init_model)

            y_pred = model.predict(X_valid)
            mse = mean_squared_error(y_valid, y_pred)

            return mse

        # Optuna hyperparameter tuning
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

        # Get the best hyperparameters
        best_params = study.best_params
        best_model = LGBMRegressor(**best_params)

        best_model.fit(X, y, eval_set=eval_set, sample_weight=train_weights, eval_sample_weight=[eval_weights])

        return best_model
