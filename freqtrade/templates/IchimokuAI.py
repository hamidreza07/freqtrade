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


class IchimokuAI(IStrategy):


    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"prediction": {"color": "blue"}},
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

    std_dev_multiplier_buy = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], default=1.25, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], space="sell", default=1.25, optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
      

        dataframe["%-high"] =dataframe["high"]
        dataframe["%-close"] =  dataframe["close"]
        dataframe["%-low"] = dataframe["low"] 
        dataframe["%-open"] = dataframe["open"]


        return dataframe
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:

        return 3.0
    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        
        dataframe[f'%-EMA9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe[f'%-EMA25'] = ta.EMA(dataframe, timeperiod=26)
        dataframe[f'%-EMA52'] = ta.EMA(dataframe, timeperiod=52)
        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        ichi = ichimoku(dataframe)
        dataframe['&-tenkan'] = ichi['tenkan_sen']
        dataframe['&-kijun'] = ichi['kijun_sen']
        dataframe['&-senkou_a'] = ichi['senkou_span_a']
        dataframe['&-senkou_b'] = ichi['senkou_span_b']



        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [
              # Condition 1: Leading span b > Leading span a
        df['&-senkou_b'] > df['&-senkou_a'],

        # Condition 2: Leading Span B should be a constant number for eleven consecutive candles

        ta.COUNT(df['&-senkou_b'].shift(1) == df['&-senkou_b'].shift(2), timeperiod=11) == 11,

        # Condition 4: The Tenkan Sen line crossed above the Kijun Sen line
        df['&-tenkan'] > df['&-kijun'],
            ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")



        enter_short_conditions = [
            df['&-senkou_b'] < df['&-senkou_a'],

            # Condition 2: Leading Span B should be a constant number for eleven consecutive candles

            ta.COUNT(df['&-senkou_b'].shift(1) == df['&-senkou_b'].shift(2), timeperiod=11) == 11,
            
            # Condition 4: The Tenkan Sen line crossed above the Kijun Sen line
            df['&-tenkan'] < df['&-kijun'],
            ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

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