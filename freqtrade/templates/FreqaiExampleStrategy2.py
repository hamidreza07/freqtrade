import logging
from functools import reduce
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy
from datetime import datetime

logger = logging.getLogger(__name__)

class FreqaiExampleStrategy2(IStrategy):
    """
    Completely new strategy with the same function and class names.
    """

    minimal_roi = {"0": 0.01, "40":-1}

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
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        Access metadata such as the current pair with:

        `metadata["pair"]`

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .max()
            / dataframe["close"]
            - 1
        )
        # Classifiers are typically set up with strings as targets:
        # df['&s-up_or_down'] = np.where( df["close"].shift(-100) >
        #                                 df["close"], 'up', 'down')

        # If user wishes to use multiple targets, they can add more by
        # appending more columns with '&'. User should keep in mind that multi targets
        # requires a multioutput prediction model such as
        # freqai/prediction_models/CatboostRegressorMultiTarget.py,
        # freqtrade trade --freqaimodel CatboostRegressorMultiTarget

        # df["&-s_range"] = (
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .max()
        #     -
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .min()
        # )
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
        Completely different entry trend logic.
        """

        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > df["%-sma_50"]*0.01,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < df["%-sma_50"],
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < df[f"%-ema_200"] ,
            ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > df[f"%-ema_200"]*0.01 ,
            ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1


        return df
    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Completely different trade entry confirmation logic.
        """

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        # logger.info(f'dataframeeee:{df}')
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.01)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.01)):
                return False

        return True
