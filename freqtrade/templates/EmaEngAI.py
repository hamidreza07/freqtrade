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

    buy_EMA_parameter = DecimalParameter(0.00, 1, default= 0.3, optimize=True)

    trade_trigger = CategoricalParameter(["can_short", "can_long","can_both"],default="can_both", space='buy', optimize=True)
    if trade_trigger.value=='can_long':
        can_short = False
    else:
        can_short = True
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:


        dataframe["%-high"] = dataframe["high"]
        dataframe["%-close"] = dataframe['close']
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
              dataframe[f'&-ema{val}'] = ta.EMA(dataframe, timeperiod=val)


        return dataframe
    


    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)


        dataframe["new_ema"] = (
            (dataframe[f'&-ema{self.buy_range_EMA.value}'] - (dataframe[f'&-ema{self.buy_range_EMA.value}'] * self.buy_EMA_parameter.value/ 100))
            )
        


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:


        if self.trade_trigger.value=="can_both":
             df = self.trading_both(df)
        elif self.trade_trigger.value == "can_long":
             df = self.trading_long(df)
        elif self.trade_trigger.value == "can_short":
             df = self.trading_short(df)


   



        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
                        (df["do_predict"] == 1),

        df["close"] > df[f'&-ema{self.sell_range_EMA.value}']
        ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
                        (df["do_predict"] == 1),

        df["close"] < df[f'&-ema{self.sell_range_EMA.value}']

        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
    



    def trading_long(self,df:DataFrame):
    

        df.loc[
            (df["do_predict"] == 1)&
            (df["close"].shift(1) < df["open"].shift(1))& 
            (df["open"] <= df["close"].shift(1))&  # Open <= previous close
            (df["close"] > df["open"].shift(1))&
            (df["close"] <= df["new_ema"] ),  # Close <= EMA - (EMA * 0.3)/100

                  # Close > previous open,
            'enter_long'] = 1
        return df
         
    def trading_short(self,df:DataFrame):
        df.loc[
                        (df["do_predict"] == 1)&

            (df["close"].shift(1) > df["open"].shift(1))& 
            (df["open"] >= df["close"].shift(1))&  # Open >= previous close
            (df["close"] < df["open"].shift(1))&
            (df["close"] >= df["new_ema"]),  # Close <= EMA - (EMA * 0.3)/100

                  # Close > previous open,
            'enter_short'] = 1
   
        return df
         
    def trading_both(self,df:DataFrame):
         df = self.trading_long(df)
         df = self.trading_short(df)
         return df
    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True         