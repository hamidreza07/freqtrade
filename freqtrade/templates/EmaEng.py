import talib.abstract as ta
import numpy as np  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
import talib

class EmaEng(IStrategy):

    timeframe = '5m'


    # ROI table:
    minimal_roi = {
       "0": 0.03,
      "58": 0.02,
      "175": 0.01



    }
    # # Stoploss:
    stoploss = -0.05
 

    buy_range_EMA = IntParameter(2, 50, default=17, optimize=True)
    sell_range_EMA = IntParameter(2, 50, default=47, optimize=True)

    buy_EMA_parameter = DecimalParameter(0.00, 1, default= 0.043, optimize=True)

    trade_trigger = CategoricalParameter(["can_short", "can_long","can_both"],default="can_long", space='buy', optimize=True)
    if trade_trigger.value=='can_long':
        can_short = False
    else:
        can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in list(set(list(self.buy_range_EMA.range) + list(self.sell_range_EMA.range))):
              dataframe[f'ema{val}'] = ta.EMA(dataframe, timeperiod=val)


        


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        if self.trade_trigger.value=="can_both":
             df = self.trading_both(df)
        if self.trade_trigger.value == "can_long":
             df = self.trading_long(df)
        if self.trade_trigger.value == "can_short":
             df = self.trading_short(df)


   



        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
        df["close"] > df[f'ema{self.sell_range_EMA.value}']
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
        df["close"] < df[f'ema{self.sell_range_EMA.value}']

        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
    



    def trading_long(self,df:DataFrame):
    

        df.loc[
            (df["close"].shift(1) < df["open"].shift(1))& 
            (df["open"] <= df["close"].shift(1))&  # Open <= previous close
            (df["close"] > df["open"].shift(1))&
            (df["close"] <= (df[f'ema{self.buy_range_EMA.value}'] - (df[f'ema{self.buy_range_EMA.value}'] * self.buy_EMA_parameter.value / 100))),  # Close <= EMA - (EMA * 0.3)/100

                  # Close > previous open,
            'enter_long'] = 1
        return df
         
    def trading_short(self,df:DataFrame):
        df.loc[
            (df["close"].shift(1) > df["open"].shift(1))& 
            (df["open"] >= df["close"].shift(1))&  # Open >= previous close
            (df["close"] < df["open"].shift(1))&
            (df["close"] >= (df[f'ema{self.buy_range_EMA.value}'] - (df[f'ema{self.buy_range_EMA.value}'] * self.buy_EMA_parameter.value/ 100))),  # Close <= EMA - (EMA * 0.3)/100

                  # Close > previous open,
            'enter_short'] = 1
   
        return df
         
    def trading_both(self,df:DataFrame):
         df = self.trading_long(df)
         df = self.trading_short(df)
         return df
         