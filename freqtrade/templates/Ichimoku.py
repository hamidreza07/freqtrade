import logging
from functools import reduce
from typing import Dict,Optional
from technical.indicators import ichimoku
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy
from datetime import datetime

logger = logging.getLogger(__name__)


class Ichimoku(IStrategy):


    minimal_roi = {"0": 0.01}



    process_only_new_candles = True
    stoploss = -0.99
    trailing_stop_loss = False
    can_short = True

    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:

        return 3.0
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['EMA'] = ta.EMA(dataframe, timeperiod=200)
        ichi = ichimoku(dataframe)
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']


        return dataframe




    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
            ((df['close'] > df["EMA"])&
            (df['close'] > df['senkou_a'])&
            (df['close'].shift(26)>df['senkou_a'])&
            (qtpylib.crossed_above(df['tenkan'],df['kijun'])) )|
            (
            (df['close'] > df["EMA"])&
            (df['tenkan']>df['kijun']))&
            (qtpylib.crossed_below(df['close'],df['senkou_a'])) 
            ),

            'enter_long'] = 1

        df.loc[
            (
            ((df['close'] < df["EMA"])&
            (df['close'] < df['senkou_a'])&
            (df['close'].shift(26)< df['senkou_a'])&
            (qtpylib.crossed_below(df['tenkan'],df['kijun']) ))|
            (
            (df['close'] <  df["EMA"])&
            (df['tenkan'] < df['kijun'])&
            (qtpylib.crossed_above(df['close'],df['senkou_a'])) )
            ),
            'enter_short'] = 1

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

