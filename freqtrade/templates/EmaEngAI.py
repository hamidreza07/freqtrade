import talib.abstract as ta
import numpy as np  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
from typing import Dict
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
import talib
import logging
logger = logging.getLogger(__name__)

class EmaEngAI(IStrategy):

    timeframe = '5m'


    # # # ROI table:
    # minimal_roi = {
    #    "0": 0.01


    # }
    # # Stoploss:
    stoploss = -0.03
 
    can_short = True
    buy_range_EMA = IntParameter(2, 70, default=50, optimize=True)
    sell_range_EMA = IntParameter(2, 70, default=50, optimize=True)

    buy_EMA_parameter = DecimalParameter(0.00, 1, default= 0.3, optimize=True)

    EMA_trigger = CategoricalParameter([True,False],default=True, space='buy', optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:


        dataframe["%-high"] = dataframe["high"]
        dataframe["%-low"] = dataframe['low']
        dataframe["%-open"] = dataframe['open']



        return dataframe
    
    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        for val in list(set(list(self.buy_range_EMA.range) + list(self.sell_range_EMA.range))):
              dataframe[f'&-ema{val}_1'] = (ta.EMA(dataframe, timeperiod=val)).shift(-30)
              dataframe[f'&-ema{val}_2'] = (ta.EMA(dataframe, timeperiod=val)).shift(-40)
              dataframe[f'&-ema{val}_3'] = (ta.EMA(dataframe, timeperiod=val)).shift(-50)


        return dataframe
    


    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['ema_now'] =ta.EMA(dataframe, timeperiod=20)
        if self.EMA_trigger.value:
            dataframe["new_ema"] = (dataframe[f'ema_now'] - (dataframe[f'ema_now'] * self.buy_EMA_parameter.value / 100) )
            
        else:

            dataframe["new_ema"] = (dataframe[f'&-ema{self.buy_range_EMA.value}_2'] - (dataframe[f'&-ema{self.buy_range_EMA.value}_2'] * self.buy_EMA_parameter.value / 100) )
            
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            ((df["do_predict"] == 1)&
            (df["close"].shift(1) < df["open"].shift(1))& 
            (df["open"] <= df["close"].shift(1))&  # Open <= previous close
            (df["close"] > df["open"].shift(1))&
            (df["close"] <= df["new_ema"] )&
            (df[f'&-ema{self.buy_range_EMA.value}_1']>df['ema_now'])&
            (df[f'&-ema{self.buy_range_EMA.value}_2']>df['ema_now'])&
            (df[f'&-ema{self.buy_range_EMA.value}_3']>df['ema_now'])),
            

            'enter_long'] = 1
        df.loc[
            ((df["do_predict"] == 1)&

            (df["close"].shift(1) > df["open"].shift(1))& 
            (df["open"] >= df["close"].shift(1))&  
            (df["close"] < df["open"].shift(1))&
            (df["close"] >= df["new_ema"])& 
            (df[f'&-ema{self.buy_range_EMA.value}_1']<df['ema_now'] )&
            (df[f'&-ema{self.buy_range_EMA.value}_2']<df['ema_now'])&
            (df[f'&-ema{self.buy_range_EMA.value}_3']<df['ema_now'])),


                 
            'enter_short'] = 1
   

   



        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            ((df["do_predict"] == 1)&

                (df["close"] > df[f'&-ema{self.sell_range_EMA.value}_1'])|
                (df["close"] > df[f'&-ema{self.sell_range_EMA.value}_2'])|
                (df["close"] > df[f'&-ema{self.sell_range_EMA.value}_3'])),
            

                  # Close > previous open,
            'exit_long'] = 1


        df.loc[
            ((df["do_predict"] == 1)&

                (df["close"] < df[f'&-ema{self.sell_range_EMA.value}_1'])|
                (df["close"] < df[f'&-ema{self.sell_range_EMA.value}_2'])|
                (df["close"] < df[f'&-ema{self.sell_range_EMA.value}_3'])),
            

                  # Close > previous open,
            'exit_short'] = 1

        return df
    





    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

