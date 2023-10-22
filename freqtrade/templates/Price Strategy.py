import talib.abstract as ta
import numpy as np  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter


class EmaEng(IStrategy):

    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.1,
        "69": 0.15,

    }

    # Stoploss:
    stoploss = - 0.01









    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=1)

        


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            # Condition 1: The last 6 candles should be lower than the previous one.
            df['close'].iloc[-1] < df['close'].iloc[-2],
            df['close'].iloc[-2] < df['close'].iloc[-3],
            df['close'].iloc[-3] < df['close'].iloc[-4],
            df['close'].iloc[-4] < df['close'].iloc[-5],
            df['close'].iloc[-5] < df['close'].iloc[-6],

            # Condition 2: The distance between the 7th and 6th > 2x the distance between the 5th and 4th.
            (df['close'].iloc[-6] - df['close'].iloc[-7]) > 2 * (df['close'].iloc[-4] - df['close'].iloc[-5]),

            # Condition 3: The 8th candle should be Bullish (you can define your own condition for a Bullish candle).
            # For example, if you have a condition for a Bullish candle, you can replace the following line:
            df['ema'].iloc[-1] > df['sma'].iloc[-1],

            # Condition 4: The difference between the price of the 8th candle and the price of the 6th candle should be NEGATIVE.
            (df['close'].iloc[-6] - df['close'].iloc[-8]) < 0
        ]




        if (enter_long_conditions) :
                df.loc[
                    reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
                ] = (1, "long")

        enter_short_conditions = [
            # Condition 1: The last 6 candles should be higher than the previous one.
            df['close'].iloc[-1] > df['close'].iloc[-2],
            df['close'].iloc[-2] > df['close'].iloc[-3],
            df['close'].iloc[-3] > df['close'].iloc[-4],
            df['close'].iloc[-4] > df['close'].iloc[-5],
            df['close'].iloc[-5] > df['close'].iloc[-6],

            # Condition 2: The distance between the 7th and 6th > 2x the distance between the 5th and 4th.
            (df['close'].iloc[-6] - df['close'].iloc[-7]) > 2 * (df['close'].iloc[-4] - df['close'].iloc[-5]),

            # Condition 3: The 8th candle should be Bearish (define your condition for a Bearish candle).
            # Replace the following line with your Bearish condition:
            df['ema'].iloc[-1] < df['sma'].iloc[-1],

            # Condition 4: The difference between the price of the 8th candle and the price of the 6th candle should be POSITIVE.
            (df['close'].iloc[-6] - df['close'].iloc[-8]) > 0
        ]


        if enter_short_conditions :
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [


        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
