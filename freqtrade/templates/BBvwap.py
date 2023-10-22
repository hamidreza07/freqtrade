import logging
import talib.abstract as ta
import talib
import numpy as np  # noqa
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from pandas import DataFrame
# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
logger = logging.getLogger(__name__)

class BBvwap(IStrategy):

    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.1,
        "69": 0.15,

    }

    # Stoploss:
    stoploss = -0.99



    entry_rsi_upper = IntParameter(low=15, high=70, default=55, space='buy', optimize=True, load=True)
    entry_rsi_lower= IntParameter(low=15, high=70, default=45, space='buy', optimize=True, load=True)



    Exit_rsi_upper = IntParameter(low=30, high=100, default=90, space='sell', optimize=True, load=True)
    Exit_rsi_lower = IntParameter(low=30, high=100, default=10, space='sell', optimize=True, load=True)





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe,timeperiod=14)



        length = 20
        mult = 2.0   
        basis = ta.SMA(dataframe, timeperiod=20)

        # Calculate standard deviation
        dev = mult * dataframe['close'].rolling(window=length).std()

        # Calculate upper and lower Bollinger Bands
        upper = basis + dev
        lower = basis - dev

        # Calculate Bollinger Band Ratio (bbr)
        bbr = (dataframe['close'] - lower) / (upper - lower)

        # Add the Bollinger Band Ratio to your dataframeFrame
        dataframe['bbr'] = bbr
        # Calculate the VWAP
     
        dataframe['VWAP'] = qtpylib.rolling_vwap(dataframe)
        

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["rsi"] < self.entry_rsi_lower.value,
            df['bbr'] < 0,
        ]
        # # Check for the last 15 consecutive candles with OHLC > VWAP
        last_15_candles = df.iloc[-15:]
        last_15_candles_ohlc_greater_than_vwap = last_15_candles[(last_15_candles['open']>last_15_candles['VWAP'])& (last_15_candles['low']>last_15_candles['VWAP'])
                                                                 & (last_15_candles['close']>last_15_candles['VWAP'])& (last_15_candles['high']>last_15_candles['VWAP'])]
        logger.info(last_15_candles[['VWAP','high','low','close','open']])
        if len(last_15_candles_ohlc_greater_than_vwap)==15:
            enter_long_conditions.append(True)
        else:
            enter_long_conditions.append(False)


        if (enter_long_conditions):
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["rsi"] > self.entry_rsi_upper.value,
            df['bbr'] > 1,
        ]

        last_15_candles_ohlc_less_than_vwap = last_15_candles[(last_15_candles['open']<last_15_candles['VWAP'])& (last_15_candles['low']<last_15_candles['VWAP'])
                                                                 & (last_15_candles['close']<last_15_candles['VWAP'])& (last_15_candles['high']<last_15_candles['VWAP'])]
        
        if len(last_15_candles_ohlc_less_than_vwap)==15:
            enter_short_conditions.append(True)
        else:
            enter_short_conditions.append(False)
        if (enter_short_conditions) :
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df



    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
        (qtpylib.crossed_above(df['rsi'], self.Exit_rsi_upper.value))
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            (qtpylib.crossed_below(df['rsi'], self.Exit_rsi_lower.value))
        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
