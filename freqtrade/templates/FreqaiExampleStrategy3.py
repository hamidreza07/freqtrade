import logging
from functools import reduce
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy
from datetime import datetime

logger = logging.getLogger(__name__)

class FreqaiExampleStrategy3(IStrategy):
    """
    Completely new strategy with the same function and class names.
    """

    minimal_roi = {"0": 0.01, "60":0.01,"120":-1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"prediction": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 40
    can_short = True

    std_dev_multiplier_buy = CategoricalParameter(
        [0.5, 0.75, 1.0, 1.25], default=1.0, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.5, 0.75, 1.0, 1.25], space="sell", default=1.0, optimize=True)



    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                    metadata: Dict, **kwargs) -> DataFrame:
        """
        Completely different feature engineering logic.
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        
        # Calculate Stochastic Oscillator
        stoch = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)
        dataframe["%-stochastic-k"] = stoch["slowk"]
        dataframe["%-stochastic-d"] = stoch["slowd"]
        
        dataframe["%-macd"] = ta.MACD(dataframe)['macd']

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 3.0

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        Access metadata such as the current pair/timeframe with:

        `metadata["pair"]` `metadata["tf"]`

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        Access metadata such as the current pair with:

        `metadata["pair"]`

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Set a target with a continuous range of numbers based on a moving average crossover.
        """
        # Calculate short-term moving average (e.g., 3-period MA)
        short_term_period = 3
        dataframe['short_term_ma'] = dataframe['close'].rolling(window=short_term_period).mean()

        # Calculate long-term moving average (e.g., 10-period MA)
        long_term_period = 10
        dataframe['long_term_ma'] = dataframe['close'].rolling(window=long_term_period).mean()
        short_term_ma = dataframe["short_term_ma"]  # Replace with your actual short-term MA
        long_term_ma = dataframe["long_term_ma"]    # Replace with your actual long-term MA

        # Calculate the difference between short-term and long-term MAs
        ma_crossover_diff = short_term_ma - long_term_ma

        # Assign the difference as the target, scaling it for better readability
        dataframe["&-ma_crossover_target"] = ma_crossover_diff * 100

        return dataframe



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All indicators must be populated by feature_engineering_*() functions

        # the model will return all labels created by user in `set_freqai_targets()`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["%-sma_50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["%-ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Modify entry trend logic to use the combined trailing target.

        :param df: DataFrame with strategy data
        :param metadata: Metadata of the current pair
        :return: Modified DataFrame with entry trend information
        """
        enter_long_conditions = [
            df["do_predict"] == 1 , df["&-ma_crossover_target"] > 10
        ]
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")


        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-ma_crossover_target"] < -10 ,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Modify exit trend logic to use the combined trailing target.

        :param df: DataFrame with strategy data
        :param metadata: Metadata of the current pair
        :return: Modified DataFrame with exit trend information
        """
        enter_long_conditions = [
            df["do_predict"] == 1 , df["&-ma_crossover_target"] < 5,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-ma_crossover_target"] > -5 ,
            ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df


    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])


