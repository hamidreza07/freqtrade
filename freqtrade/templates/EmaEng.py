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
    stoploss = -0.01





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f'ema'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=1)

        


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
        (df["open"] <= df["close"].shift(1)),  # Open <= previous close
        (df["close"] > df["open"].shift(1)),   # Close > previous open
        (df["close"] <= (df["ema"] - (df["ema"] * 0.3 / 100)))  # Close <= EMA - (EMA * 0.3)/100
    
        ]


        if (enter_long_conditions) :
                df.loc[
                    reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
                ] = (1, "long")

        enter_short_conditions = [
        (df["open"] >= df["close"].shift(1)),  # Open <= previous close
        (df["close"] < df["open"].shift(1)),   # Close > previous open
        (df["close"] >= (df["ema"] - (df["ema"] * 0.3 / 100)))  # Close <= EMA - (EMA * 0.3)/100
    
            ]

        if enter_short_conditions :
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
        df["close"] > df["ema"]
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
        df["close"] < df["ema"]

        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
