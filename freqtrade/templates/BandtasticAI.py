import talib.abstract as ta
import numpy as np  # noqa
import logging
from typing import Dict

import numpy as np  # noqa
import pandas as pd  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
logger = logging.getLogger(__name__)


# Optimized With Sharpe Ratio and 1 year data
# 199/40000:  30918 trades. 18982/3408/8528 Wins/Draws/Losses. Avg profit   0.39%. Median profit   0.65%. Total profit  119934.26007495 USDT ( 119.93%). Avg duration 8:12:00 min. Objective: -127.60220

class BandtasticAI(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '3m'

    # ROI table:
    minimal_roi = {
        "0": 0.162,
        "69": 0.097,
        "229": 0.061,
        "566": 0
    }
    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = True

    # Stoploss:
    stoploss = -0.345

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.058
    trailing_only_offset_is_reached = False

    # Hyperopt Buy Parameters
    buy_fastema = IntParameter(low=1, high=236, default=211, space='buy', optimize=True, load=True)
    buy_slowema = IntParameter(low=1, high=126, default=364, space='buy', optimize=True, load=True)
    buy_rsi = IntParameter(low=15, high=70, default=52, space='buy', optimize=True, load=True)
    buy_mfi = IntParameter(low=15, high=70, default=30, space='buy', optimize=True, load=True)

    buy_rsi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_mfi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_ema_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_trigger = CategoricalParameter(["bb_lower1", "bb_lower2", "bb_lower3", "bb_lower4"], default="bb_lower1", space="buy")

    # Hyperopt Sell Parameters
    sell_fastema = IntParameter(low=1, high=365, default=7, space='sell', optimize=True, load=True)
    sell_slowema = IntParameter(low=1, high=365, default=6, space='sell', optimize=True, load=True)
    sell_rsi = IntParameter(low=30, high=100, default=57, space='sell', optimize=True, load=True)
    sell_mfi = IntParameter(low=30, high=100, default=46, space='sell', optimize=True, load=True)

    sell_rsi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_mfi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=True)
    sell_ema_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_trigger = CategoricalParameter(["sell-bb_upper1", "sell-bb_upper2", "sell-bb_upper3", "sell-bb_upper4"], default="sell-bb_upper2", space="sell")
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        dataframe["%-high"] = dataframe["high"]
        dataframe["%-close"] = dataframe["close"]
        dataframe["%-low"] = dataframe["low"] 
        dataframe["%-open"] = dataframe["open"]


        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe
    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
   

        # Bollinger Bands 1,2,3 and 4
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['&s-bb_lowerband1'] = bollinger1['lower']
        # dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['&s-bb_upperband1'] = bollinger1['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['&s-bb_lowerband2'] = bollinger2['lower']
        # dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['&s-bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['&s-bb_lowerband3'] = bollinger3['lower']
        # dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['&s-bb_upperband3'] = bollinger3['upper']

        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['&s-bb_lowerband4'] = bollinger4['lower']
        # dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['&s-bb_upperband4'] = bollinger4['upper']
        return dataframe   
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)


        # Build EMA rows - combine all ranges to a single set to avoid duplicate calculations.
        for period in set(
                list(self.buy_fastema.range)
                + list(self.buy_slowema.range)
                + list(self.sell_fastema.range)
                + list(self.sell_slowema.range)
            ):
            dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,  
            df['rsi'] < self.buy_rsi.value,
            df['rsi'] < self.buy_rsi.value ,
            df['mfi'] < self.buy_mfi.value,
            df[f'EMA_{self.buy_fastema.value}'] > df[f'EMA_{self.buy_slowema.value}']]



        # TRIGGERS
    
        if self.buy_trigger.value == 'bb_lower1':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband1'])
        if self.buy_trigger.value == 'bb_lower2':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband2'])
        if self.buy_trigger.value == 'bb_lower3':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband3'])
        if self.buy_trigger.value == 'bb_lower4':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband4'])


        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
        enter_short_conditions = [
            df["do_predict"] == 1,  
            df['rsi'] > self.buy_rsi.value,
            df['rsi'] > self.buy_rsi.value ,
            df['mfi'] > self.buy_mfi.value,
            df[f'EMA_{self.buy_fastema.value}'] > df[f'EMA_{self.buy_slowema.value}']]



        # TRIGGERS
    
        if self.buy_trigger.value == 'bb_lower1':
            enter_short_conditions.append(df["close"] > df['&s-bb_lowerband1'])
        if self.buy_trigger.value == 'bb_lower2':
            enter_short_conditions.append(df["close"] > df['&s-bb_lowerband2'])
        if self.buy_trigger.value == 'bb_lower3':
            enter_short_conditions.append(df["close"] > df['&s-bb_lowerband3'])
        if self.buy_trigger.value == 'bb_lower4':
            enter_short_conditions.append(df["close"] > df['&s-bb_lowerband4'])



        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        pass
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

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True