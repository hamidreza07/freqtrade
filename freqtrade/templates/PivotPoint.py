import logging
import talib.abstract as ta
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import numpy as np  # noqa
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from pandas import DataFrame
# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, merge_informative_pair, DecimalParameter, IntParameter
logger = logging.getLogger(__name__)
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
class PivotPoint(IStrategy):
    buy_params = {
    'Support1_parameter' : .382,
    'Support2_parameter' : .618,
    'Resistance1_parameter' : .382,
    'Resistance2_parameter' :.618,
    'Support1_Woodie_parameter': 2,
    'Support3_Woodie_parameter' :2,
    'Resistance1_Woodie_parameter':2,
    'Resistance3_Woodie_parameter':2,
    'trade_trigger' :"can_both",
    'calculation_trigger' :"Fibonacci"
    }
   


    sell_params = {"time_trigger":"1d"}

     

    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.02
         ,"69": 0.01,




    }
    # Stoploss:
    stoploss = -0.05
    plot_config = {
        "main_plot": {         
                "Pivot_1d": {"Pivot_1d": {"color": "navy"},},
                
                
                
                "Support1_1d": {"Support1_1d": {"color": "brown"},},

                "Support2_1d": {"Support2_1d": {"color": "yellow"},},

                "Support3_1d": {"Support3_1d": {"color": "black"},},



                "Resistance1_1d": {"Resistance1_1d": {"color": "olive"},},

                "Resistance2_1d": {"Resistance2_1d": {"color": "blue"},},
                
                "Resistance3_1d": {"Resistance3_1d": {"color": "teal"},},

       
        },
    }
    can_short = True

    Support1_parameter = DecimalParameter(0.00, 1, default=.382, space="buy", optimize=True, load=True)
    Support2_parameter = DecimalParameter(0.00, 1, default=.618, space="buy", optimize=True, load=True)
    Resistance1_parameter = DecimalParameter(0.00, 1, default=.382, space="buy", optimize=True, load=True)
    Resistance2_parameter = DecimalParameter(0.00, 1, default=.618, space="buy", optimize=True, load=True)
    Support1_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True, load=True)
    Support3_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True, load=True)
    Resistance1_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True, load=True)
    Resistance3_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True, load=True)
    trade_trigger = CategoricalParameter(["can_short", "can_long","can_both"],default="can_both", space='buy', optimize=True)
    calculation_trigger = CategoricalParameter(["Woodie", "Fibonacci"],default="Fibonacci", space='buy', optimize=True)


    time_trigger = CategoricalParameter(["1w", "1d"],default="1d", space='sell', optimize=True)



    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        informative_pairs +=[(pair, '1w') for pair in pairs]
        
        return informative_pairs


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.time_trigger.value=="1w":
            self.inf_tf = '1w'
        if self.time_trigger.value=="1d":
            self.inf_tf = '1d'
            # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_tf)

        
        if self.calculation_trigger.value=="Woodie":
            informative['Pivot'] = (informative['high'] + informative['low'] + informative['close']) / 3
            informative['Support1'] = (self.Support1_Woodie_parameter.value * informative['Pivot']) - informative['high']   
            informative['Support2'] = informative['Pivot'] - (informative['high'] - informative['low'])
            informative['Support3'] = informative['low'] - self.Support3_Woodie_parameter.value * (informative['high'] - informative['Pivot'])
            informative['Resistance1'] = (self.Resistance1_Woodie_parameter.value * informative['Pivot']) - informative['low']
            informative['Resistance2'] = informative['Pivot'] + (informative['high'] - informative['low'])
            informative['Resistance3'] = informative['high'] + self.Resistance3_Woodie_parameter.value  * (informative['Pivot'] - informative['low'])



        if self.calculation_trigger.value=="Fibonacci":

            informative['Pivot'] = (informative['high'] + informative['low'] + informative['close']) / 3
            informative['Support1'] = informative['Pivot'] - self.Support1_parameter.value * (informative['high'] - informative['low'])
            informative['Support2'] = informative['Pivot'] - self.Support2_parameter.value * (informative['high'] - informative['low'])
            informative['Support3'] = informative['Pivot'] - (informative['high'] - informative['low'])
            informative['Resistance1'] = informative['Pivot'] + self.Resistance1_parameter.value * (informative['high'] - informative['low'])
            informative['Resistance2'] = informative['Pivot'] + self.Resistance2_parameter.value * (informative['high'] - informative['low'])
            informative['Resistance3'] = informative['Pivot'] + (informative['high'] - informative['low'])

        informative['date'] = pd.to_datetime(informative['date'])


        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_tf, ffill=True)







        return dataframe




    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['condition_entry_long_1'] = qtpylib.crossed_above(df['close'],df[f'Support2_{self.inf_tf}'])
        df['condition_entry_long_2'] = qtpylib.crossed_above (df['close'] , df[f'Support1_{self.inf_tf}'])
        df['condition_entry_long_3'] = qtpylib.crossed_above (df['close'] , df[f'Pivot_{self.inf_tf}'])
        df['condition_entry_long_4'] = qtpylib.crossed_above(df['close'] , df[f'Resistance1_{self.inf_tf}'])
        df['condition_entry_long_5'] = qtpylib.crossed_above (df['close'] , df[f'Resistance2_{self.inf_tf}'])


        df['condition_entry_short_1'] = qtpylib.crossed_below (df['close'] , df[f'Resistance2_{self.inf_tf}'])
        df['condition_entry_short_2'] = qtpylib.crossed_below  (df['close'] , df[f'Resistance1_{self.inf_tf}'])
        df['condition_entry_short_3'] = qtpylib.crossed_below  (df['close'] , df[f'Pivot_{self.inf_tf}'])
        df['condition_entry_short_4'] = qtpylib.crossed_below (df['close'] , df[f'Support1_{self.inf_tf}'])
        df['condition_entry_short_5'] = qtpylib.crossed_below (df['close'] , df[f'Support2_{self.inf_tf}'])

        if self.trade_trigger.value=="can_both":
             df = self.trading_both(df)
        if self.trade_trigger.value == "can_long":
             df = self.trading_long(df)
        if self.trade_trigger.value == "can_short":
             df = self.trading_short(df)




        return df
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dataframe['condition_exit_long_1'] = (qtpylib.crossed_above (dataframe['close'] , dataframe[f'Support1_{self.inf_tf}']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Support3_{self.inf_tf}'])) 
        dataframe['condition_exit_long_2'] = (qtpylib.crossed_above (dataframe['close'] , dataframe[f'Pivot_{self.inf_tf}']))| (qtpylib.crossed_below(dataframe['close'],dataframe[f'Support2_{self.inf_tf}']))
        dataframe['condition_exit_long_3'] = (qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance1_{self.inf_tf}']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Support1_{self.inf_tf}']))
        dataframe['condition_exit_long_4'] = (qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance2_{self.inf_tf}']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Pivot_{self.inf_tf}']))
        dataframe['condition_exit_long_5'] = (qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance3_{self.inf_tf}']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Resistance1_{self.inf_tf}']))

   
       
        dataframe['condition_exit_short_1'] =  (qtpylib.crossed_below (dataframe['close'] , dataframe[f'Resistance1_{self.inf_tf}']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance3_{self.inf_tf}']))
        dataframe['condition_exit_short_2'] =  (qtpylib.crossed_below (dataframe['close'] , dataframe[f'Pivot_{self.inf_tf}']))|( qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance2_{self.inf_tf}']))
        dataframe['condition_exit_short_3'] =  (qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support1_{self.inf_tf}']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance1_{self.inf_tf}']))
        dataframe['condition_exit_short_4'] =  (qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support2_{self.inf_tf}']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance1_{self.inf_tf}']))
        dataframe['condition_exit_short_5'] =  (qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support3_{self.inf_tf}']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Support1_{self.inf_tf}']))
        last_candle = dataframe.iloc[-1].squeeze()


        last_candle['date_stamp'] = pd.to_datetime(last_candle['date']) 
        H = last_candle['date_stamp'].hour  
        M = last_candle['date_stamp'].minute  
        W = last_candle['date_stamp'].dayofweek
        if self.time_trigger.value=="1w": 
              time_condition = (W==0)
        if self.time_trigger.value=="1d":
            time_condition =(H==0) & (M==5)

        if trade.enter_tag =="long_signal_1" :
            if time_condition:
                  return f'exit_{trade.enter_tag}'  
            elif last_candle['condition_exit_long_1']:
                    return f'exit_{trade.enter_tag}'

        if trade.enter_tag =="long_signal_2" :
            if time_condition:
                 return f'exit_{trade.enter_tag}'   
            elif last_candle['condition_exit_long_2']:
                    return f'exit_{trade.enter_tag}'



        if trade.enter_tag =="long_signal_3" :
            if time_condition:
                  return f'exit_{trade.enter_tag}'             
            elif last_candle['condition_exit_long_3']:
                    return f'exit_{trade.enter_tag}'



        if trade.enter_tag =="long_signal_4" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif last_candle['condition_exit_long_4']:
                    return f'exit_{trade.enter_tag}'


        if trade.enter_tag =="long_signal_5" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif last_candle['condition_exit_long_5']:
                    return f'exit_{trade.enter_tag}'


                


        if trade.enter_tag =="short_signal_1" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif last_candle['condition_exit_short_1']: 
                    return f'exit_{trade.enter_tag}'

        if trade.enter_tag =="short_signal_2" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif last_candle['condition_exit_short_2']:
                    return f'exit_{trade.enter_tag}'



        if trade.enter_tag =="short_signal_3" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif  last_candle['condition_exit_short_3']:
                    return f'exit_{trade.enter_tag}'



        if trade.enter_tag =="short_signal_4" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif  last_candle['condition_exit_short_4']:
                    return f'exit_{trade.enter_tag}'


        if trade.enter_tag =="short_signal_5" :
            if time_condition:
                  return f'exit_{trade.enter_tag}' 
            elif last_candle['condition_exit_short_5']:
                    return f'exit_{trade.enter_tag}'


        return None


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        

        return df
    


    def trading_long(self,df:DataFrame):
        df.loc[df['condition_entry_long_1'], ["enter_long", "enter_tag"]] =  (1, "long_signal_1")
        df.loc[df['condition_entry_long_2'], ["enter_long", "enter_tag"]] =  (1, "long_signal_2")
        df.loc[df['condition_entry_long_3'], ["enter_long", "enter_tag"]] =  (1, "long_signal_3")
        df.loc[df['condition_entry_long_4'], ["enter_long", "enter_tag"]] =  (1, "long_signal_4")
        df.loc[ df['condition_entry_long_5'], ["enter_long", "enter_tag"]] = (1, "long_signal_5")
        return df
         
    def trading_short(self,df:DataFrame):
        df.loc[df['condition_entry_short_1'], ["enter_short", "enter_tag"]] =  (1, "short_signal_1")
        df.loc[df['condition_entry_short_2'], ["enter_short", "enter_tag"]] =  (1, "short_signal_2")
        df.loc[df['condition_entry_short_3'], ["enter_short", "enter_tag"]] =  (1, "short_signal_3")
        df.loc[df['condition_entry_short_4'], ["enter_short", "enter_tag"]] =  (1, "short_signal_4")
        df.loc[ df['condition_entry_short_5'], ["enter_short", "enter_tag"]] = (1, "short_signal_5")
        return df
         
    def trading_both(self,df:DataFrame):
         df = self.trading_long(df)
         df = self.trading_short(df)
         return df
         