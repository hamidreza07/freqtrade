import logging
import talib.abstract as ta
import talib
import numpy as np  # noqa
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from pandas import DataFrame
# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, merge_informative_pair, IntParameter, RealParameter
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

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        
        return informative_pairs


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        inf_tf = '1d'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['Pivot'] = (informative['high'] + informative['low'] + informative['close']) / 3

        informative['Support1'] = (2 * informative['Pivot']) - informative['high']
        informative['Support2'] = informative['Pivot'] - (informative['high'] - informative['low'])
        informative['Resistance1'] = (2 * informative['Pivot']) - informative['low']
        informative['Resistance2'] = informative['Pivot'] + (informative['high'] - informative['low'])
        informative['date'] = pd.to_datetime(informative['date'])


        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)







        return dataframe




    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['condition_entry_long_1'] = (df['close'].shift(1) < df['Support2_1d'].shift(1)) & (df['close'] >= df['Support2_1d'])
        df['condition_entry_long_2'] = (df['close'].shift(1) < df['Support1_1d'].shift(1)) & (df['close'] >= df['Support1_1d'])
        df['condition_entry_long_3'] = (df['close'].shift(1) < df['Pivot_1d'].shift(1)) & (df['close'] >= df['Pivot_1d'])
        df['condition_entry_long_4'] = (df['close'].shift(1) < df['Resistance1_1d'].shift(1)) & (df['close'] >= df['Resistance1_1d'])
        df['condition_entry_long_5'] = (df['close'].shift(1) < df['Resistance2_1d'].shift(1)) & (df['close'] >= df['Resistance2_1d'])


        df['condition_entry_short_1'] = (df['close'].shift(1) > df['Resistance2_1d'].shift(1)) & (df['close'] <= df['Resistance2_1d'])
        df['condition_entry_short_2'] = (df['close'].shift(1) > df['Resistance1_1d'].shift(1)) & (df['close'] <= df['Resistance1_1d'])
        df['condition_entry_short_3'] = (df['close'].shift(1) > df['Pivot_1d'].shift(1)) & (df['close'] <= df['Pivot_1d'])
        df['condition_entry_short_4'] = (df['close'].shift(1) > df['Support1_1d'].shift(1)) & (df['close'] <= df['Support1_1d'])
        df['condition_entry_short_5'] = (df['close'].shift(1) > df['Support2_1d'].shift(1)) & (df['close'] <= df['Support2_1d'])

        enter_long_conditions = [

       ( (df['close'].shift(1) < df['Support2_1d'].shift(1)) & (df['close'] >= df['Support2_1d'])),
     (   (df['close'].shift(1) < df['Support1_1d'].shift(1)) & (df['close'] >= df['Support1_1d'])),
      (  (df['close'].shift(1) < df['Pivot_1d'].shift(1)) & (df['close'] >= df['Pivot_1d'])),
(        (df['close'].shift(1) < df['Resistance1_1d'].shift(1)) & (df['close'] >= df['Resistance1_1d'])),
(        (df['close'].shift(1) < df['Resistance2_1d'].shift(1)) & (df['close'] >= df['Resistance2_1d']) )
            ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x | y , enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
        enter_short_conditions = [
        (df['close'].shift(1) > df['Resistance2_1d'].shift(1)) & (df['close'] <= df['Resistance2_1d']),
        (df['close'].shift(1) > df['Resistance1_1d'].shift(1)) & (df['close'] <= df['Resistance1_1d']),
        (df['close'].shift(1) > df['Pivot_1d'].shift(1)) & (df['close'] <= df['Pivot_1d']),
        (df['close'].shift(1) > df['Support1_1d'].shift(1)) & (df['close'] <= df['Support1_1d']),
        (df['close'].shift(1) > df['Support2_1d'].shift(1)) & (df['close'] <= df['Support2_1d'])]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x | y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")


        return df

    



    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['condition_exit_long_1'] = (df['condition_entry_long_1']==True) &(df['close'].shift(1) < df['Support2_1d'].shift(1)) & (df['close'] >= df['Support2_1d'])
        df['condition_exit_long_2'] = (df['condition_entry_long_2']==True) & (df['close'].shift(1) < df['Pivot_1d'].shift(1)) & (df['close'] >= df['Pivot_1d'])
        df['condition_exit_long_3'] = (df['condition_entry_long_3']==True) & (df['close'].shift(1) < df['Resistance1_1d'].shift(1)) & (df['close'] >= df['Resistance1_1d'])
        df['condition_exit_long_4'] = (df['condition_entry_long_4']==True ) & (df['close'].shift(1) < df['Resistance2_1d'].shift(1)) & (df['close'] >= df['Resistance2_1d'])
   
       
        df['condition_exit_short_1'] = (df['condition_entry_short_1']) &  (df['close'].shift(1) > df['Resistance1_1d'].shift(1)) & (df['close'] <= df['Resistance1_1d'])
        df['condition_exit_short_2'] = (df['condition_entry_short_2']) & (df['close'].shift(1) > df['Pivot_1d'].shift(1)) & (df['close'] <= df['Pivot_1d'])
        df['condition_exit_short_3'] = (df['condition_entry_short_3'] ) & (df['close'].shift(1) > df['Support1_1d'].shift(1)) & (df['close'] <= df['Support1_1d'])
        df['condition_exit_short_4'] = (df['condition_entry_short_4']) & (df['close'].shift(1) > df['Support2_1d'].shift(1)) & (df['close'] <= df['Support2_1d'])
        exit_long_conditions = [
           df['condition_exit_long_1'],
            df['condition_exit_long_2'] ,
            df['condition_exit_long_3'],
            df['condition_exit_long_4']
            ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x | y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            df['condition_exit_short_1'],
            df['condition_exit_short_2'] ,
            df['condition_exit_short_3'],
            df['condition_exit_short_4']
            ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x | y, exit_short_conditions), "exit_short"] = 1

        return df
