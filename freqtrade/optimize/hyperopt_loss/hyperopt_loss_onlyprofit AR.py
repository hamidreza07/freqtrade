"""
OnlyProfitHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss

from freqtrade.constants import Config

from datetime import datetime
from freqtrade.data.metrics import calculate_sharpe


class OnlyProfitHyperArima(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation takes only absolute profit into account, not looking at any other indicator.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               config: Config,min_date: datetime, max_date: datetime,*args, **kwargs,) -> float:
        """
        Objective function, returns smaller number for better results.
        """
        p = config['model_training_parameters']['p']
        q = config['model_training_parameters']['q']
        d = config['model_training_parameters']['d']

        sharp_ratio1 = calculate_sharpe(results, min_date, max_date, p)
        sharp_ratio2 = calculate_sharpe(results, min_date, max_date, q)
        sharp_ratio3 = calculate_sharpe(results, min_date, max_date, d)


        return 1-(sharp_ratio1*sharp_ratio2*sharp_ratio3)
