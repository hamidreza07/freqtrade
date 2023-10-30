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
    Pivot = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], default=1.25, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], space="sell", default=1.25, optimize=True)




    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate pivot point levels
        dataframe['Pivot'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['Support1'] = (2 * dataframe['Pivot']) - dataframe['high']
        dataframe['Support2'] = dataframe['Pivot'] - (dataframe['high'] - dataframe['low'])
        dataframe['Support3'] = dataframe['low'] - 2 * (dataframe['high'] - dataframe['Pivot'])
        dataframe['Resistance1'] = (2 * dataframe['Pivot']) - dataframe['low']
        dataframe['Resistance2'] = dataframe['Pivot'] + (dataframe['high'] - dataframe['low'])
        dataframe['Resistance3'] = dataframe['high'] + 2 * (dataframe['Pivot'] - dataframe['low'])


        dataframe['condition_entry_long_5'] = qtpylib.crossed_above(dataframe['close'], dataframe['Resistance2'])
        dataframe['condition_entry_long_4'] = (qtpylib.crossed_above(dataframe['close'], dataframe['Resistance1'])) & (dataframe['close']<dataframe['Resistance2'])
        dataframe['condition_entry_long_1'] = (qtpylib.crossed_above(dataframe['close'], dataframe['Pivot'])) & (dataframe['close']<dataframe['Resistance1'])
        dataframe['condition_entry_long_2'] = (qtpylib.crossed_above(dataframe['close'], dataframe['Support1'])) & (dataframe['close']<dataframe['Pivot'])
        dataframe['condition_entry_long_3'] = (qtpylib.crossed_above(dataframe['close'], dataframe['Support2']) )& (dataframe['close']<dataframe['Support1'])



        dataframe['condition_entry_short_5'] = (qtpylib.crossed_below(dataframe['close'], dataframe['Resistance2'])) & (dataframe['close'] > dataframe['Resistance1'])
        dataframe['condition_entry_short_4'] = (qtpylib.crossed_below(dataframe['close'], dataframe['Resistance1'])) & (dataframe['close'] > dataframe['Pivot'])
        dataframe['condition_entry_short_1'] = (qtpylib.crossed_below(dataframe['close'], dataframe['Pivot'])) & (dataframe['close']> dataframe['Support1'])
        dataframe['condition_entry_short_2'] = (qtpylib.crossed_below(dataframe['close'], dataframe['Support1'])) & (dataframe['close'] > dataframe['Support2'])
        dataframe['condition_entry_short_3'] = qtpylib.crossed_below(dataframe['close'], dataframe['Support2'])



        return dataframe




    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
     
                df['condition_entry_long_5']|
                df['condition_entry_long_4']|
                df['condition_entry_long_1']|
                df['condition_entry_long_3'] 
            ),
            'enter_long'] = 1



        df.loc[
            (
                df['condition_entry_short_5']|
                df['condition_entry_short_4']|
                df['condition_entry_short_1']|
                df['condition_entry_short_2'] 
                                        ),
            'enter_short'] = 1
        return df

    

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['condition_exit_long_5'] = (df['close'] >= df['Resistance2'])
        df['condition_exit_long_4'] = (df['close'] >= df['Resistance1'])

        df['condition_exit_long_1'] = (df['close'] >= df['Pivot'])
        df['condition_exit_long_2'] = (df['close'] >= df['Support1'])
        df.loc[
            (
            (df['condition_entry_long_1']==True & (df['condition_exit_long_4']==True )) |
            (df['condition_entry_long_2']==True & (df['condition_exit_long_1']==True )) |
            (df['condition_entry_long_3']==True & (df['condition_exit_long_2']==True )) |
            (df['condition_entry_long_4']==True & (df['condition_exit_long_5']==True ) )
            ),

            'exit_long'] = 1
        
        df['condition_exit_short_5'] = (df['close']<= df['Resistance2'])
        df['condition_exit_short_4'] = (df['close']<= df['Resistance1'])
        df['condition_exit_short_1'] = (df['close']<= df['Pivot'])
        df['condition_exit_short_2'] = (df['close']<= df['Support1'])
        df['condition_exit_short_3'] = (df['close']<= df['Support2'])

        # Define the exit condition_exits based on your requirements
        df.loc[
            (
            (df['condition_entry_short_1']==True & (df['condition_exit_short_2']==True )) |
            (df['condition_entry_short_2']==True & (df['condition_exit_short_3']==True )) |
            (df['condition_entry_short_4']==True & (df['condition_exit_short_1']==True )) |
            (df['condition_entry_short_5']==True & (df['condition_exit_short_4']==True ) )
            ),

            'exit_short'] = 1
        
        return df
