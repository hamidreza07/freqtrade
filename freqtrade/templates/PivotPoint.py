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

class PivotPoint(IStrategy):

    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.1,
        "69": 0.2,




    }

    # Stoploss:
    stoploss = -0.05



    entry_rsi_upper = IntParameter(low=15, high=70, default=55, space='buy', optimize=True, load=True)
    entry_rsi_lower= IntParameter(low=15, high=70, default=45, space='buy', optimize=True, load=True)



    Exit_rsi_upper = IntParameter(low=30, high=100, default=90, space='sell', optimize=True, load=True)
    Exit_rsi_lower = IntParameter(low=30, high=100, default=10, space='sell', optimize=True, load=True)





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate pivot point levels
        dataframe['Pivot'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['Support1'] = (2 * dataframe['Pivot']) - dataframe['high']
        dataframe['Support2'] = dataframe['Pivot'] - (dataframe['high'] - dataframe['low'])
        dataframe['Support3'] = dataframe['low'] - 2 * (dataframe['high'] - dataframe['Pivot'])
        dataframe['Resistance1'] = (2 * dataframe['Pivot']) - dataframe['low']
        dataframe['Resistance2'] = dataframe['Pivot'] + (dataframe['high'] - dataframe['low'])
        dataframe['Resistance3'] = dataframe['high'] + 2 * (dataframe['Pivot'] - dataframe['low'])



        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df['intersect_Pivot'] = qtpylib.crossed_above(df['close'],df['Pivot'])
        df['intersect_Support1'] = qtpylib.crossed_above(df['close'],df['Support1'])
        df['intersect_Support2'] = qtpylib.crossed_above(df['close'],df['Support2'])
        df['intersect_Resistance1'] = qtpylib.crossed_above(df['close'],df['Resistance1'])
        df['intersect_Resistance2'] = qtpylib.crossed_above(df['close'],df['Resistance2'])

        enter_long_conditions = [
            df['intersect_Pivot']== True| 
            df['intersect_Support1']== True|
            df['intersect_Support2'] == True|
             df['intersect_Resistance1']== True|
            df['intersect_Resistance2']== True

        ]



        if (enter_long_conditions):
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        df['intersect_Pivot'] = qtpylib.crossed_below(df['close'],df['Pivot'])
        df['intersect_Support1'] = qtpylib.crossed_below(df['close'],df['Support1'])
        df['intersect_Support2'] = qtpylib.crossed_below(df['close'],df['Support2'])
        df['intersect_Resistance1'] = qtpylib.crossed_below(df['close'],df['Resistance1'])
        df['intersect_Resistance2'] = qtpylib.crossed_below(df['close'],df['Resistance2'])

        enter_short_conditions = [
             df['intersect_Pivot']== True| 
            df['intersect_Support1']== True|
            df['intersect_Support2'] == True|
             df['intersect_Resistance1']== True|
            df['intersect_Resistance2']== True
        ]


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