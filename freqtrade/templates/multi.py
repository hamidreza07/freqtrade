# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from typing import Dict, Union
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import logging

from freqtrade.strategy.informative_decorator import informative
logger = logging.getLogger(__name__)

class multi2(IStrategy):
   
    INTERFACE_VERSION = 3

    timeframe = '5m'

    minimal_roi = {
        "0": 0.1,
    }
    
    informative_timeframe = '1h'
    informative_timeframe2 = '15m'
   


    stoploss = -0.35
    trailing_stop = False   

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    
   
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get the 14 day rsi
        dataframe[f'rsi'] = ta.RSI(dataframe, timeperiod=14)
        # Calculate MACD
        macd_default = (12, 26, 9)
  
        macd = ta.MACD(dataframe, *macd_default)
        dataframe[f'macd'] = macd['macd']
        dataframe[f'macd_signal'] = macd['macdsignal']


        # Calculate Stochastic Oscillator
        stoch = ta.STOCHF(dataframe, 5,3,3)
        dataframe[f'fastk'] = stoch['fastk']
        dataframe[f'fastd'] = stoch['fastd']

        return dataframe
   


    @informative('15m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f'rsi'] = ta.RSI(dataframe, timeperiod=14)

        macd_default = (12, 26, 9)
        macd = ta.MACD(dataframe, *macd_default)
        dataframe[f'macd'] = macd['macd']
        dataframe[f'macd_signal'] = macd['macdsignal']


        stoch = ta.STOCHF(dataframe, 5,3,3)
        dataframe[f'fastk'] = stoch['fastk']
        dataframe[f'fastd'] = stoch['fastd']
        return dataframe



    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f'rsi'] = ta.RSI(dataframe, timeperiod=14)

        macd_default = (12, 26, 9)
        macd = ta.MACD(dataframe, *macd_default)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']


        stoch = ta.STOCHF(dataframe, 5, 3, 3)
        dataframe['fastk'] = stoch['fastk']
        dataframe['fastd'] = stoch['fastd']



        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < 50) &
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['fastk'] > dataframe['fastd']) &
            (dataframe['fastk'] < 70) &
            (dataframe[f'rsi_{self.informative_timeframe}'] < 46) &
            (dataframe[f'macd_{self.informative_timeframe}'] > dataframe[f'macd_signal_{self.informative_timeframe}']) &
            (dataframe[f'fastk_{self.informative_timeframe}'] > dataframe[f'fastd_{self.informative_timeframe}']) &
            (dataframe[f'fastk_{self.informative_timeframe}'] < 72) &
            (dataframe[f'rsi_{self.informative_timeframe2}'] < 45) &
            (dataframe[f'macd_{self.informative_timeframe2}'] > dataframe[f'macd_signal_{self.informative_timeframe2}'])&
            (dataframe[f'fastk_{self.informative_timeframe2}'] > dataframe[f'fastd_{self.informative_timeframe2}']) &
            (dataframe[f'fastk_{self.informative_timeframe2}'] < 80)
            , 'enter_long'] = 1
        
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            
                (
                   
                    dataframe['sma_50'] <= dataframe['sma_200'] 

                ),
                'exit_long'] = 1
       
        

        
        return dataframe
        