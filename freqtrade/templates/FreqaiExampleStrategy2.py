import logging
from functools import reduce
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy
from datetime import datetime

logger = logging.getLogger(__name__)

class FreqaiExampleStrategy2(IStrategy):
    """
    Completely new strategy with the same function and class names.
    """

    minimal_roi = {"0": 0.2, "30": -1}

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
    use_take_profit = True

    startup_candle_count: int = 60
    can_short = True

    std_dev_multiplier_buy = CategoricalParameter(
        [0.5, 0.75, 1.0, 1.25], default=1.0, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.5, 0.75, 1.0, 1.25], space="sell", default=1.0, optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
        """
        Completely different feature engineering logic.
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-stochastic-period"] = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)
        dataframe["%-macd"] = ta.MACD(dataframe)['macd']

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Completely different leverage logic.
        """

        
        return 3.0

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Completely different basic feature engineering logic.
        """
        dataframe["%-sma-50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Completely different standard feature engineering logic.
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Completely different target setting logic.
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .max()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Completely different indicator population logic.
        """
        dataframe["%-ema-10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["%-bb_upperband"] = ta.BBANDS(dataframe)['upperband']
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Completely different entry trend logic.
        """
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["close"] > df["%-sma-50"],
            ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["close"] < df["%-sma-50"],
            ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Completely different exit trend logic.
        """
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["close"] < df["%-bb_upperband"],
            ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            df["do_predict"] == 1,
            df["close"] > df["%-bb_upperband"],
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
        """
        Completely different trade entry confirmation logic.
        """

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.01)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.01)):
                return False

        return True
