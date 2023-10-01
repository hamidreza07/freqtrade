import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union,Dict

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime

from functools import reduce

class UTBotAI(IStrategy):
    
    INTERFACE_VERSION = 3

  

    minimal_roi = {"0": 0.01, "40":-1}


    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0  

    timeframe = '3m'

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 40
    can_short = True
    key_value_l = IntParameter(low=1, high=10, default=1, space='buy', optimize=True, load=True)
    key_value_s = IntParameter(low=1, high=10, default=1, space='buy', optimize=True, load=True)

    atr_period_l = IntParameter(low=1, high=20, default=3, space='buy', optimize=True, load=True)
    atr_period_s = IntParameter(low=1, high=20, default=3, space='buy', optimize=True, load=True)

    ema_period_l = IntParameter(low=50, high=500, default=200, space='buy', optimize=True, load=True)
    ema_period_s = IntParameter(low=50, high=500, default=200, space='buy', optimize=True, load=True)

    ema_period_l_exit = IntParameter(low=20, high=200, default=50, space='sell', optimize=True, load=True)
    ema_period_s_exit = IntParameter(low=20, high=200, default=50, space='sell', optimize=True, load=True)

    volume_check = IntParameter(low=5, high=50, default=20, space='buy', optimize=True, load=True)
    volume_check_exit = IntParameter(low=5, high=50, default=20, space='sell', optimize=True, load=True)

    volume_check_s = IntParameter(low=5, high=50, default=20, space='buy', optimize=True, load=True)
    volume_check_exit_s = IntParameter(low=5, high=50, default=20, space='sell', optimize=True, load=True)

    adx_long_min = DecimalParameter(low=20, high=40, decimals=2, default=25.00, space="buy", optimize=True, load=True)
    adx_long_max = DecimalParameter(low=60, high=80, decimals=2, default=75.00, space="buy", optimize=True, load=True)

    adx_short_min = DecimalParameter(low=20, high=40, decimals=2, default=25.00, space="buy", optimize=True, load=True)
    adx_short_max = DecimalParameter(low=60, high=80, decimals=2, default=75.00, space="buy", optimize=True, load=True)


    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def UT_Alert(self, dataframe, key_value=1, atr_period=3, ema_period=200):

        xATR = np.array(ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=atr_period))
        nLoss = key_value * xATR
        src = dataframe['close']

        xATRTrailingStop = np.zeros(len(dataframe))
        xATRTrailingStop[0] = src[0] - nLoss[0]

        mask_1 = (src > np.roll(xATRTrailingStop, 1)) & (np.roll(src, 1) > np.roll(xATRTrailingStop, 1))
        mask_2 = (src < np.roll(xATRTrailingStop, 1)) & (np.roll(src, 1) < np.roll(xATRTrailingStop, 1))
        mask_3 = src > np.roll(xATRTrailingStop, 1)

        xATRTrailingStop = np.where(mask_1, np.maximum(np.roll(xATRTrailingStop, 1), src - nLoss), xATRTrailingStop)
        xATRTrailingStop = np.where(mask_2, np.minimum(np.roll(xATRTrailingStop, 1), src + nLoss), xATRTrailingStop)
        xATRTrailingStop = np.where(mask_3, src - nLoss, xATRTrailingStop)

        mask_buy = (np.roll(src, 1) < xATRTrailingStop) & (src > np.roll(xATRTrailingStop, 1))
        mask_sell = (np.roll(src, 1) > xATRTrailingStop) & (src < np.roll(xATRTrailingStop, 1))

        pos = np.zeros(len(dataframe))
        pos = np.where(mask_buy, 1, pos)
        pos = np.where(mask_sell, -1, pos)
        pos[~((pos == 1) | (pos == -1))] = 0

        ema = np.array(ta.EMA(dataframe['close'], timeperiod=ema_period))

        buy_condition_utbot = (xATRTrailingStop > ema) & (pos > 0) & (src > ema)
        sell_condition_utbot = (xATRTrailingStop < ema) & (pos < 0) & (src < ema)

        trend = np.where(buy_condition_utbot, 1, np.where(sell_condition_utbot, -1, 0))
        trend = np.array(trend)

        dataframe['trend'] = trend

        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
        dataframe['%-close'] = dataframe["close"]
        dataframe['%-open'] = dataframe["open"]
        dataframe['%-high'] = dataframe["high"]
        dataframe['%-low'] = dataframe["low"]

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

        dataframe['&-s_adx'] = ta.ADX(dataframe)

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


        

        L_optimize_trend_alert  = self.UT_Alert(dataframe=dataframe, key_value= self.key_value_l.value, atr_period= self.atr_period_l.value, ema_period=self.ema_period_l.value)
        dataframe['trend_l'] = L_optimize_trend_alert['trend']

        S_optimize_trend_alert  = self.UT_Alert(dataframe=dataframe, key_value= self.key_value_s.value, atr_period= self.atr_period_s.value, ema_period=self.ema_period_s.value)
        dataframe['trend_s'] = S_optimize_trend_alert['trend']


        dataframe['ema_l'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period_l_exit.value)
        dataframe['ema_s'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period_s_exit.value)

        dataframe['volume_mean'] = dataframe['volume'].rolling(self.volume_check.value).mean().shift(1)
        dataframe['volume_mean_exit'] = dataframe['volume'].rolling(self.volume_check_exit.value).mean().shift(1)

        dataframe['volume_mean_s'] = dataframe['volume'].rolling(self.volume_check_s.value).mean().shift(1)
        dataframe['volume_mean_exit_s'] = dataframe['volume'].rolling(self.volume_check_exit_s.value).mean().shift(1)
        
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        enter_long_conditions = [
            df["do_predict"] == 1,
             (df['&-s_adx'] > self.adx_long_min.value) ,
            (df['&-s_adx'] < self.adx_long_max.value) ,
            (df['trend_l'] > 0) ,
            (df['volume'] > df['volume_mean']) ,
            (df['volume'] > 0)
        ]
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            (df['&-s_adx'] > self.adx_short_min.value) , 
            (df['&-s_adx'] < self.adx_short_max.value) , 
            (df['trend_s'] < 0) ,
            (df['volume'] > df['volume_mean_s'])]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df["do_predict"] == 1,
            (df['close'] < df['ema_l']) ,
            (df['volume'] > df['volume_mean_exit'])]
        
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1
        
        exit_short_conditions = [
            df["do_predict"] == 1,
            # (df['close'] > df['high'].shift(self.sell_shift_short.value)) &
            (df['close'] > df['ema_s']) &
            (df['volume'] > df['volume_mean_exit_s'])]

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