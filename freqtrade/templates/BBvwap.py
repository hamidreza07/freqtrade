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

class BBvwap(IStrategy):

    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.1,
        "69": 0.15,
        "119":0.05,
        "179":0,
        "239":-1



    }

    # Stoploss:
    stoploss = -0.05



    entry_rsi_upper = IntParameter(low=15, high=70, default=55, space='buy', optimize=True, load=True)
    entry_rsi_lower= IntParameter(low=15, high=70, default=45, space='buy', optimize=True, load=True)



    Exit_rsi_upper = IntParameter(low=30, high=100, default=90, space='sell', optimize=True, load=True)
    Exit_rsi_lower = IntParameter(low=30, high=100, default=10, space='sell', optimize=True, load=True)





    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe,timeperiod=14)



        length = 20
        mult = 2.0   
        basis = ta.SMA(dataframe, timeperiod=20)

        # Calculate standard deviation
        dev = mult * dataframe['close'].rolling(window=length).std()

        # Calculate upper and lower Bollinger Bands
        upper = basis + dev
        lower = basis - dev

        # Calculate Bollinger Band Ratio (bbr)
        bbr = (dataframe['close'] - lower) / (upper - lower)

        # Add the Bollinger Band Ratio to your dataframeFrame
        dataframe['bbr'] = bbr
        # Calculate the VWAP
     
        # dataframe['VWAP'] = qtpylib.rolling_vwap(dataframe)
                # Calculate the VWAP
        dataframe['TypicalPrice'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['TypicalPriceVolume'] = dataframe['TypicalPrice'] * dataframe['volume']
        dataframe['CumulativeTPV'] = dataframe['TypicalPriceVolume'].cumsum()
        dataframe['CumulativeVolume'] = dataframe['volume'].cumsum()
        dataframe['VWAP'] = dataframe['CumulativeTPV'] / dataframe['CumulativeVolume']

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df['open_close_low_high_gt_vwap'] = np.where(
        (df['open'] > df['VWAP']) &
        (df['close'] > df['VWAP']) &
        (df['low'] > df['VWAP']) &
        (df['high'] > df['VWAP']), 1, 0
    )

        # Initialize a variable for the cumulative sum
        cumulative_sum_high = 0



        # Create a new column for the cumulative sum
        df['cumulative_sum_column_high'] = 0

        for index, row in df.iterrows():
            if row['open_close_low_high_gt_vwap'] == 0 :
                cumulative_sum_high = 0  # Reset the cumulative sum
            else:
            # Update the cumulative sum
                cumulative_sum_high += 1

            # Update the cumulative sum column in the DataFrame
            df.at[index, 'cumulative_sum_column_high'] = cumulative_sum_high
        enter_long_conditions = [
            df["rsi"] < self.entry_rsi_lower.value,
            df['bbr'] < 0,
            df['cumulative_sum_column_high']>=15
        ]



        if (enter_long_conditions):
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")



        df['open_close_low_high_lt_vwap'] = np.where(
        (df['open'] < df['VWAP']) &
        (df['close'] < df['VWAP']) &
        (df['low'] < df['VWAP']) &
        (df['high'] < df['VWAP']), 1, 0
    )

        # Initialize a variable for the cumulative sum
        cumulative_sum_low = 0

        df['cumulative_sum_high_low'] = 0

        for index, row in df.iterrows():
            if row['open_close_low_high_lt_vwap'] == 0 :
                cumulative_sum_low = 0  # Reset the cumulative sum
            else:
                 # Update the cumulative sum
                cumulative_sum_low += 1

            # Update the cumulative sum column in the DataFrame
            df.at[index, 'cumulative_sum_high_low'] = cumulative_sum_low
        enter_short_conditions = [
            df["rsi"] > self.entry_rsi_upper.value,
            df['bbr'] > 1,
            df['cumulative_sum_high_low']>=15
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