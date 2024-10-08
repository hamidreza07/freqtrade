import logging
from functools import reduce
from typing import Dict,Optional
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade

from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy,DecimalParameter,IntParameter
from datetime import datetime

from freqtrade.strategy.strategy_helper import merge_informative_pair

logger = logging.getLogger(__name__)


class PivotPointAI(IStrategy):

    minimal_roi = {"0": 0.05}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "s_close": {"prediction": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 40
    can_short = True

    buy_range_EMA = IntParameter(2, 70, default=20, optimize=True)
    sell_range_EMA = IntParameter(2, 70, default=20, optimize=True)
    Support1_parameter = DecimalParameter(0.00, 1, default=.382, space="buy", optimize=True)
    Support2_parameter = DecimalParameter(0.00, 1, default=.618, space="buy", optimize=True)
    Resistance1_parameter = DecimalParameter(0.00, 1, default=.382, space="buy", optimize=True)
    Resistance2_parameter = DecimalParameter(0.00, 1, default=.618, space="buy", optimize=True)

    Support1_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True)
    Support3_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True)


    Resistance1_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True)
    Resistance3_Woodie_parameter = IntParameter(low=1, high=20, default=2, space='buy', optimize=True)

    time_trigger = CategoricalParameter(["1w", "1d"],default="1d", space='buy', optimize=True)
    trade_trigger = CategoricalParameter(["can_short", "can_long","can_both"],default="can_both", space='buy', optimize=True)


    calculation_trigger = CategoricalParameter(["Woodie", "Fibonacci"],default="Fibonacci", space='buy', optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame,period,
                                       metadata: Dict, **kwargs) -> DataFrame:



        dataframe["%-high"] = dataframe['high']
        dataframe["%-low"]  = dataframe['low']
        dataframe["%-open"] = dataframe['open']

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )


        return dataframe

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
       
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        informative_pairs +=[(pair, '1w') for pair in pairs]
        
        return informative_pairs
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        for val in list(set(list(self.buy_range_EMA.range) + list(self.sell_range_EMA.range))):
              dataframe[f'&-ema{val}_1'] = (ta.EMA(dataframe, timeperiod=val)).shift(-30)
              dataframe[f'&-ema{val}_2'] = (ta.EMA(dataframe, timeperiod=val)).shift(-40)
              dataframe[f'&-ema{val}_3'] = (ta.EMA(dataframe, timeperiod=val)).shift(-50)



        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe = self.freqai.start(dataframe, metadata, self)
        if self.time_trigger.value=="1w":
            self.inf_tf = '1w'
        if self.time_trigger.value=="1d":
            self.inf_tf = '1d'
            # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_tf)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_tf, ffill=True)
        if self.calculation_trigger.value=="Woodie":
            
            dataframe['Pivot'] = (dataframe[f'high_{self.inf_tf}'] + dataframe[f'low_{self.inf_tf}'] + dataframe[f'close_{self.inf_tf}']) / 3
            dataframe['Support1'] = (self.Support1_Woodie_parameter.value * dataframe['Pivot']) - dataframe[f'high_{self.inf_tf}']   
            dataframe['Support2'] = dataframe['Pivot'] - (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Support3'] = dataframe[f'low_{self.inf_tf}'] - self.Support3_Woodie_parameter.value * (dataframe[f'high_{self.inf_tf}'] - dataframe['Pivot'])
            dataframe['Resistance1'] = (self.Resistance1_Woodie_parameter.value * dataframe['Pivot']) - dataframe[f'low_{self.inf_tf}']
            dataframe['Resistance2'] = dataframe['Pivot'] + (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Resistance3'] = dataframe[f'high_{self.inf_tf}'] + self.Resistance3_Woodie_parameter.value  * (dataframe['Pivot'] - dataframe[f'low_{self.inf_tf}'])



        if self.calculation_trigger.value=="Fibonacci":

            dataframe['Pivot'] = (dataframe[f'high_{self.inf_tf}'] + dataframe[f'low_{self.inf_tf}'] + dataframe[f'close_{self.inf_tf}']) / 3
            dataframe['Support1'] = dataframe['Pivot'] - self.Support1_parameter.value * (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Support2'] = dataframe['Pivot'] - self.Support2_parameter.value * (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Support3'] = dataframe['Pivot'] - (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Resistance1'] = dataframe['Pivot'] + self.Resistance1_parameter.value * (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Resistance2'] = dataframe['Pivot'] + self.Resistance2_parameter.value * (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])
            dataframe['Resistance3'] = dataframe['Pivot'] + (dataframe[f'high_{self.inf_tf}'] - dataframe[f'low_{self.inf_tf}'])

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['condition_entry_long_1'] = qtpylib.crossed_above(df['close'],df[f'Support2']) & (df[f'&-ema{self.buy_range_EMA.value}_1']>df[f'Support1'])& (df[f'&-ema{self.buy_range_EMA.value}_2']>df[f'Support1'])& (df[f'&-ema{self.buy_range_EMA.value}_3']>df[f'Support1'])
        df['condition_entry_long_2'] = qtpylib.crossed_above (df['close'] ,df[f'Support1']) & (df[f'&-ema{self.buy_range_EMA.value}_1']>df[f'Pivot'])& (df[f'&-ema{self.buy_range_EMA.value}_2']>df[f'Pivot'])& (df[f'&-ema{self.buy_range_EMA.value}_3']>df[f'Pivot'])
        df['condition_entry_long_3'] = qtpylib.crossed_above (df['close'] ,df[f'Pivot']) & (df[f'&-ema{self.buy_range_EMA.value}_1']>df[f'Resistance1'])& (df[f'&-ema{self.buy_range_EMA.value}_2']>df[f'Resistance1'])& (df[f'&-ema{self.buy_range_EMA.value}_3']>df[f'Resistance1'])
        df['condition_entry_long_4'] = qtpylib.crossed_above(df['close'] , df[f'Resistance1']) & (df[f'&-ema{self.buy_range_EMA.value}_1']>df[f'Resistance2'])& (df[f'&-ema{self.buy_range_EMA.value}_2']>df[f'Resistance2'])& (df[f'&-ema{self.buy_range_EMA.value}_3']>df[f'Resistance2'])
        df['condition_entry_long_5'] = qtpylib.crossed_above (df['close'] , df[f'Resistance2']) & (df[f'&-ema{self.buy_range_EMA.value}_1']>df[f'Resistance3'])& (df[f'&-ema{self.buy_range_EMA.value}_2']>df[f'Resistance3'])& (df[f'&-ema{self.buy_range_EMA.value}_3']>df[f'Resistance3'])


        df['condition_entry_short_1'] = qtpylib.crossed_below(df['close'],df[f'Support2']) & (df[f'&-ema{self.buy_range_EMA.value}_1']<df[f'Support3'])& (df[f'&-ema{self.buy_range_EMA.value}_2']<df[f'Support3'])& (df[f'&-ema{self.buy_range_EMA.value}_3']<df[f'Support3'])
        df['condition_entry_short_2'] = qtpylib.crossed_below (df['close'] ,df[f'Support1']) & (df[f'&-ema{self.buy_range_EMA.value}_1']<df[f'Support2'])& (df[f'&-ema{self.buy_range_EMA.value}_2']<df[f'Support2'])& (df[f'&-ema{self.buy_range_EMA.value}_3']<df[f'Support2'])
        df['condition_entry_short_3'] = qtpylib.crossed_below (df['close'] ,df[f'Pivot']) & (df[f'&-ema{self.buy_range_EMA.value}_1']<df[f'Support1'])& (df[f'&-ema{self.buy_range_EMA.value}_2']<df[f'Support1'])& (df[f'&-ema{self.buy_range_EMA.value}_3']<df[f'Support1'])
        df['condition_entry_short_4'] = qtpylib.crossed_below(df['close'] , df[f'Resistance1']) & (df[f'&-ema{self.buy_range_EMA.value}_1']<df[f'Pivot'])& (df[f'&-ema{self.buy_range_EMA.value}_2']<df[f'Pivot'])& (df[f'&-ema{self.buy_range_EMA.value}_3']<df[f'Pivot'])
        df['condition_entry_short_5'] = qtpylib.crossed_below (df['close'] , df[f'Resistance2']) & (df[f'&-ema{self.buy_range_EMA.value}_1']<df[f'Resistance1'])& (df[f'&-ema{self.buy_range_EMA.value}_2']<df[f'Resistance1'])& (df[f'&-ema{self.buy_range_EMA.value}_3']<df[f'Resistance1'])

        df.loc[((df['condition_entry_long_1'])&(df["do_predict"] == 1)), ["enter_long", "enter_tag"]] = (1, "long_signal_1")
        df.loc[((df['condition_entry_long_2'])&(df["do_predict"] == 1)), ["enter_long", "enter_tag"]] = (1, "long_signal_2")
        df.loc[((df['condition_entry_long_3'])&(df["do_predict"] == 1)), ["enter_long", "enter_tag"]] = (1, "long_signal_3")
        df.loc[((df['condition_entry_long_4'])&(df["do_predict"] == 1)), ["enter_long", "enter_tag"]] = (1, "long_signal_4")
        df.loc[((df['condition_entry_long_5'])&(df["do_predict"] == 1)), ["enter_long", "enter_tag"]] = (1, "long_signal_5")

        df.loc[((df['condition_entry_short_1'])&(df["do_predict"] == 1)), ["enter_short", "enter_tag"]] =  (1, "short_signal_1")
        df.loc[((df['condition_entry_short_2'])&(df["do_predict"] == 1)), ["enter_short", "enter_tag"]] =  (1, "short_signal_2")
        df.loc[((df['condition_entry_short_3'])&(df["do_predict"] == 1)), ["enter_short", "enter_tag"]] =  (1, "short_signal_3")
        df.loc[((df['condition_entry_short_4'])&(df["do_predict"] == 1)), ["enter_short", "enter_tag"]] =  (1, "short_signal_4")
        df.loc[((df['condition_entry_short_5'])&(df["do_predict"] == 1)), ["enter_short", "enter_tag"]] = (1, "short_signal_5")

        return df
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dataframe['condition_exit_long_1'] = (dataframe["do_predict"] == 1)&(qtpylib.crossed_above (dataframe['close'] , dataframe[f'Support1']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Support3'])) 
        dataframe['condition_exit_long_2'] = (dataframe["do_predict"] == 1)&(qtpylib.crossed_above (dataframe['close'] , dataframe[f'Pivot']))| (qtpylib.crossed_below(dataframe['close'],dataframe[f'Support2']))
        dataframe['condition_exit_long_3'] = (dataframe["do_predict"] == 1)&(qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance1']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Support1']))
        dataframe['condition_exit_long_4'] = (dataframe["do_predict"] == 1)&(qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance2']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Pivot']))
        dataframe['condition_exit_long_5'] = (dataframe["do_predict"] == 1)&(qtpylib.crossed_above (dataframe['close'] , dataframe[f'Resistance3']))|(qtpylib.crossed_below(dataframe['close'],dataframe[f'Resistance1']))

   
       
        dataframe['condition_exit_short_1'] =  (dataframe["do_predict"] == 1)&(qtpylib.crossed_below (dataframe['close'] , dataframe[f'Resistance1']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance3']))
        dataframe['condition_exit_short_2'] =  (dataframe["do_predict"] == 1)&(qtpylib.crossed_below (dataframe['close'] , dataframe[f'Pivot']))|( qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance2']))
        dataframe['condition_exit_short_3'] =  (dataframe["do_predict"] == 1)&(qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support1']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Resistance1']))
        dataframe['condition_exit_short_4'] =  (dataframe["do_predict"] == 1)&(qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support2']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Pivot']))
        dataframe['condition_exit_short_5'] =  (dataframe["do_predict"] == 1)&(qtpylib.crossed_below (dataframe['close'] , dataframe[f'Support3']))|(qtpylib.crossed_above(dataframe['close'],dataframe[f'Support1']))
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

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])


   
         