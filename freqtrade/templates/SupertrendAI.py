"""
Supertrend strategy:
* Description: Generate a 3 supertrend indicators for 'buy' strategies & 3 supertrend indicators for 'sell' strategies
               Buys if the 3 'buy' indicators are 'up'
               Sells if the 3 'sell' indicators are 'down'
* Author: @juankysoriano (Juan Carlos Soriano)
* github: https://github.com/juankysoriano/

*** NOTE: This Supertrend strategy is just one of many possible strategies using `Supertrend` as indicator. It should on any case used at your own risk.
          It comes with at least a couple of caveats:
            1. The implementation for the `supertrend` indicator is based on the following discussion: https://github.com/freqtrade/freqtrade-strategies/issues/30 . Concretelly https://github.com/freqtrade/freqtrade-strategies/issues/30#issuecomment-853042401
            2. The implementation for `supertrend` on this strategy is not validated; meaning this that is not proven to match the results by the paper where it was originally introduced or any other trusted academic resources
"""
from functools import reduce

import logging
from numpy.lib import math
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from technical import qtpylib
from typing import Dict
logger = logging.getLogger(__name__)
class SupertrendAI(IStrategy):
    # Buy params, Sell params, ROI, Stoploss and Trailing Stop are values generated by 'freqtrade hyperopt --strategy Supertrend --hyperopt-loss ShortTradeDurHyperOptLoss --timerange=20210101- --timeframe=1h --spaces all'
    # It's encourage you find the values that better suites your needs and risk management strategies

    INTERFACE_VERSION: int = 3
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.087,
        "372": 0.058,
        "861": 0.029,
        "2221": 0
    }

    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    timeframe = '1h'

    startup_candle_count = 18

    buy_m1 = IntParameter(1, 7, default=4)
    buy_m2 = IntParameter(1, 7, default=4)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=14)
    buy_p3 = IntParameter(7, 21, default=14)

    sell_m1 = IntParameter(1, 7, default=4)
    sell_m2 = IntParameter(1, 7, default=4)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=14)
    sell_p3 = IntParameter(7, 21, default=14)
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-high"] = dataframe["high"]
        dataframe["%-close"] = dataframe["close"]
        dataframe["%-low"] = dataframe["low"] 
        dataframe["%-open"] = dataframe["open"]


        return dataframe



    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        # cs = []
        # for multiplier in self.buy_m1.range:
        #     for period in self.buy_p1.range:
        #         cs.append(f'supertrend_1_buy_{multiplier}_{period}')
        # for multiplier in self.buy_m2.range:
        #     for period in self.buy_p2.range:
        #         cs.append(f'supertrend_2_buy_{multiplier}_{period}')
        # for multiplier in self.buy_m3.range:
        #     for period in self.buy_p3.range:
        #         cs.append(f'supertrend_3_buy_{multiplier}_{period}')
        # for multiplier in self.sell_m1.range:
        #     for period in self.sell_p1.range:
        #         cs.append(f'supertrend_1_sell_{multiplier}_{period}')

        # for multiplier in self.sell_m2.range:
        #     for period in self.sell_p2.range:
        #         cs.append(f'supertrend_2_sell_{multiplier}_{period}')
        # for multiplier in self.sell_m3.range:
        #     for period in self.sell_p3.range:
        #         cs.append(f'supertrend_3_sell_{multiplier}_{period}')
        # for multiplier in self.sell_m1.range:
        #     for period in self.sell_p1.range:
        #         cs.append(f'supertrend_1_sell_{multiplier}_{period}')

        # self.freqai.class_names = ["down", "up"]
       

        dataframe[f'&s-supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] = self.supertrend(dataframe, self.buy_m1.value, self.buy_p1.value)['STX']

        dataframe[f'&s-supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] = self.supertrend(dataframe, self.buy_m2.value, self.buy_p2.value)['STX']+"2"
 
        dataframe[f'&s-supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] = self.supertrend(dataframe, self.buy_m3.value, self.buy_p3.value)['STX']+"3"
     

        dataframe[f'&s-supertrend_1_sell_{ self.sell_m1.value}_{self.sell_p1.value}'] = self.supertrend(dataframe, self.sell_m1.value, self.sell_p1.value)['STX']+"4"

        dataframe[f'&s-supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] = self.supertrend(dataframe, self.sell_m2.value, self.sell_p2.value)['STX']+"5"
     

        dataframe[f'&s-supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] = self.supertrend(dataframe, self.sell_m3.value, self.sell_p3.value)['STX']+"6"
     
        return dataframe

    


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.freqai_info = self.config["freqai"]
       
        dataframe = self.freqai.start(dataframe, metadata, self)


        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [         
                df["do_predict"] == 1,      
                df[f'&s-supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 'up' ,
                df[f'&s-supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up2' ,
                df[f'&s-supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 'up3' , 
                df['volume'] > 0 ]
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
        enter_short_conditions = [            
                df["do_predict"] == 1,
               df[f'&s-supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 'down' ,
               df[f'&s-supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'down2' ,
               df[f'&s-supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 'down3' , # The three indicators are 'up' for the current candle
               df['volume'] > 0 # There is at least some trading volume
                ]
        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [            
               df[f'&s-supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == 'down4' ,
               df[f'&s-supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down5',
               df[f'&s-supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == 'down6' , # The three indicators are 'down' for the current candle
               df['volume'] > 0 # There is at least some trading volume
            ]

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [           
               df[f'&s-supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == 'up4' ,
               df[f'&s-supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'up5' ,
               df[f'&s-supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == 'up6' , # The three indicators are 'down' for the current candle
               df['volume'] > 0 # There is at least some trading volume
            ]
        if exit_short_conditions:

            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df



    """
        Supertrend Indicator; adapted for freqtrade
        from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })