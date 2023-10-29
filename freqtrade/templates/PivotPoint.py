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

        logger.info(qtpylib.crossed_above(df['close'],df['Pivot']))
        if qtpylib.crossed_above(df['close'],df['Pivot']):
            self.condition_long = qtpylib.crossed_above(df['close'],df['Pivot'])
        if qtpylib.crossed_above(df['close'],df['Support1']):
            self.condition_long = qtpylib.crossed_above(df['close'],df['Support1'])
        if  qtpylib.crossed_above(df['close'],df['Support2']):
            self.condition_long =  qtpylib.crossed_above(df['close'],df['Support2'])
        if qtpylib.crossed_above(df['close'],df['Resistance1']):
            self.condition_long = qtpylib.crossed_above(df['close'],df['Resistance1'])
        if qtpylib.crossed_above(df['close'],df['Resistance2']):
            self.condition_long = qtpylib.crossed_above(df['close'],df['Resistance2'])


        df.loc[(self.condition_long),'enter_long'] = 1
        
        if qtpylib.crossed_below(df['close'],df['Pivot']):
            self.condition_short = qtpylib.crossed_below(df['close'],df['Pivot'])
        if qtpylib.crossed_below(df['close'],df['Support1']):
            self.condition_short = qtpylib.crossed_below(df['close'],df['Support1'])
        if  qtpylib.crossed_below(df['close'],df['Support2']):
            self.condition_short =  qtpylib.crossed_below(df['close'],df['Support2'])
        if qtpylib.crossed_below(df['close'],df['Resistance1']):
            self.condition_short = qtpylib.crossed_below(df['close'],df['Resistance1'])
        if qtpylib.crossed_below(df['close'],df['Resistance2']):
            self.condition_short = qtpylib.crossed_below(df['close'],df['Resistance2'])
        df.loc[(self.condition_short),'enter_short'] = 1
        
        return df
    

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if self.condition_long==qtpylib.crossed_above(df['close'],df['Pivot']):
            self.condition_exit_long =  qtpylib.crossed_above(df['close'],df['Support1'])|qtpylib.crossed_above(df['close'],df['Support2'])|qtpylib.crossed_above(df['close'],df['Resistance1'])|qtpylib.crossed_above(df['close'],df['Resistance2'])

        if self.condition_long==qtpylib.crossed_above(df['close'],df['Support1']):
            self.condition_exit_long =  qtpylib.crossed_above(df['close'],df['Pivot'])|qtpylib.crossed_above(df['close'],df['Support2'])|qtpylib.crossed_above(df['close'],df['Resistance1'])|qtpylib.crossed_above(df['close'],df['Resistance2'])

        if self.condition_long==qtpylib.crossed_above(df['close'],df['Support2']):
            self.condition_exit_long =  qtpylib.crossed_above(df['close'],df['Pivot'])|qtpylib.crossed_above(df['close'],df['Support1'])|qtpylib.crossed_above(df['close'],df['Resistance1'])|qtpylib.crossed_above(df['close'],df['Resistance2'])

        if self.condition_long==qtpylib.crossed_above(df['close'],df['Resistance1']):
            self.condition_exit_long =  qtpylib.crossed_above(df['close'],df['Pivot'])|qtpylib.crossed_above(df['close'],df['Support1'])|qtpylib.crossed_above(df['close'],df['Support2'])|qtpylib.crossed_above(df['close'],df['Resistance2'])

        if self.condition_long==qtpylib.crossed_above(df['close'],df['Resistance1']):
            self.condition_exit_long =  qtpylib.crossed_above(df['close'],df['Pivot'])|qtpylib.crossed_above(df['close'],df['Support1'])|qtpylib.crossed_above(df['close'],df['Support2'])|qtpylib.crossed_above(df['close'],df['Resistance1'])

        df.loc[(self.condition_exit_long),'exit_long'] = 1

        if self.condition_short==qtpylib.crossed_below(df['close'],df['Pivot']):
            self.condition_exit_short =  qtpylib.crossed_below(df['close'],df['Support1'])|qtpylib.crossed_below(df['close'],df['Support2'])|qtpylib.crossed_below(df['close'],df['Resistance1'])|qtpylib.crossed_below(df['close'],df['Resistance2'])

        if self.condition_short==qtpylib.crossed_below(df['close'],df['Support1']):
            self.condition_exit_short =  qtpylib.crossed_below(df['close'],df['Pivot'])|qtpylib.crossed_below(df['close'],df['Support2'])|qtpylib.crossed_below(df['close'],df['Resistance1'])|qtpylib.crossed_below(df['close'],df['Resistance2'])

        if self.condition_short==qtpylib.crossed_below(df['close'],df['Support2']):
            self.condition_exit_short =  qtpylib.crossed_below(df['close'],df['Pivot'])|qtpylib.crossed_below(df['close'],df['Support1'])|qtpylib.crossed_below(df['close'],df['Resistance1'])|qtpylib.crossed_below(df['close'],df['Resistance2'])

        if self.condition_short==qtpylib.crossed_below(df['close'],df['Resistance1']):
            self.condition_exit_short =  qtpylib.crossed_below(df['close'],df['Pivot'])|qtpylib.crossed_below(df['close'],df['Support1'])|qtpylib.crossed_below(df['close'],df['Support2'])|qtpylib.crossed_below(df['close'],df['Resistance2'])

        if self.condition_short==qtpylib.crossed_below(df['close'],df['Resistance1']):
            self.condition_exit_short =  qtpylib.crossed_below(df['close'],df['Pivot'])|qtpylib.crossed_below(df['close'],df['Support1'])|qtpylib.crossed_below(df['close'],df['Support2'])|qtpylib.crossed_below(df['close'],df['Resistance1'])


        df.loc[( self.condition_exit_short ),'exit_short'] = 1

        return df