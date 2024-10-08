import logging
from functools import reduce
from typing import Dict,Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import CategoricalParameter, IStrategy,IntParameter
from datetime import datetime
logger = logging.getLogger(__name__)



# Optimized With Sharpe Ratio and 1 year data
# 199/40000:  30918 trades. 18982/3408/8528 Wins/Draws/Losses. Avg profit   0.39%. Median profit   0.65%. Total profit  119934.26007495 USDT ( 119.93%). Avg duration 8:12:00 min. Objective: -127.60220

class BandtasticAI(IStrategy):
    INTERFACE_VERSION = 2
    std_dev_multiplier_buy = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], default=1.25, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], space="sell", default=1.25, optimize=True)
    timeframe = '3m'

    # ROI table:
    minimal_roi = {
        "0": 0.162,
        "69": 0.097,
        "229": 0.061,
        "566": 0
    }

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
    stoploss = -0.345

    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = True

    # Stoploss:

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.058
    trailing_only_offset_is_reached = False

    # Hyperopt Buy Parameters
    buy_fastema = IntParameter(low=1, high=236, default=211, space='buy', optimize=True, load=True)
    buy_slowema = IntParameter(low=1, high=126, default=364, space='buy', optimize=True, load=True)
    buy_rsi = IntParameter(low=15, high=70, default=52, space='buy', optimize=True, load=True)
    buy_mfi = IntParameter(low=15, high=70, default=30, space='buy', optimize=True, load=True)

    buy_rsi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_mfi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_ema_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_trigger = CategoricalParameter(["bb_lower1", "bb_lower2", "bb_lower3", "bb_lower4"], default="bb_lower1", space="buy")

    # Hyperopt Sell Parameters
    sell_fastema = IntParameter(low=1, high=365, default=7, space='sell', optimize=True, load=True)
    sell_slowema = IntParameter(low=1, high=365, default=6, space='sell', optimize=True, load=True)
    sell_rsi = IntParameter(low=30, high=100, default=57, space='sell', optimize=True, load=True)
    sell_mfi = IntParameter(low=30, high=100, default=46, space='sell', optimize=True, load=True)

    sell_rsi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_mfi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=True)
    sell_ema_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=False)
    sell_trigger = CategoricalParameter(["sell-bb_upper1", "sell-bb_upper2", "sell-bb_upper3", "sell-bb_upper4"], default="sell-bb_upper2", space="sell")
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

        Access metadata such as the current pair/timeframe with:

        `metadata["pair"]` `metadata["tf"]`
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
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 3.0
    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        Access metadata such as the current pair with:

        `metadata["pair"]`

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
   

        # Bollinger Bands 1,2,3 and 4
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['&s-bb_lowerband1'] = bollinger1['lower']
        # dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['&s-bb_upperband1'] = bollinger1['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['&s-bb_lowerband2'] = bollinger2['lower']
        # dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['&s-bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['&s-bb_lowerband3'] = bollinger3['lower']
        # dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['&s-bb_upperband3'] = bollinger3['upper']

        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['&s-bb_lowerband4'] = bollinger4['lower']
        # dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['&s-bb_upperband4'] = bollinger4['upper']
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
            )

        return dataframe   
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)


        # Build EMA rows - combine all ranges to a single set to avoid duplicate calculations.
        for period in set(
                list(self.buy_fastema.range)
                + list(self.buy_slowema.range)
                + list(self.sell_fastema.range)
                + list(self.sell_slowema.range)
            ):
            dataframe[f'EMA_{period}'] = ta.EMA(dataframe, timeperiod=period)
        for val in self.std_dev_multiplier_buy.range:
            dataframe[f'target_roi_{val}'] = (
                dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * val
                )
        for val in self.std_dev_multiplier_sell.range:
            dataframe[f'sell_roi_{val}'] = (
                dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * val
                )
     
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,df['volume'] > 0]
        



        if self.buy_rsi_enabled.value:
            enter_long_conditions.append(df['rsi'] < self.buy_rsi.value)
        if self.buy_mfi_enabled.value:
            enter_long_conditions.append(df['mfi'] < self.buy_mfi.value)
        if self.buy_ema_enabled.value:
            enter_long_conditions.append(df[f'EMA_{self.buy_fastema.value}'] > df[f'EMA_{self.buy_slowema.value}'])

        # TRIGGERS
        if self.buy_trigger.value == 'bb_lower1':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband1'])
        if self.buy_trigger.value == 'bb_lower2':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband2'])
        if self.buy_trigger.value == 'bb_lower3':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband3'])
        if self.buy_trigger.value == 'bb_lower4':
            enter_long_conditions.append(df["close"] < df['&s-bb_lowerband4'])


        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
        enter_short_conditions = [
            df["do_predict"] == 1,  df['volume'] > 0]
 
        if self.sell_rsi_enabled.value:
            enter_short_conditions.append(df['rsi'] > self.sell_rsi.value)
        if self.sell_mfi_enabled.value:
            enter_short_conditions.append(df['mfi'] > self.sell_mfi.value)
        if self.sell_ema_enabled.value:
            enter_short_conditions.append(df[f'EMA_{self.sell_fastema.value}'] < df[f'EMA_{self.sell_slowema.value}'])

        # TRIGGERS
        if self.sell_trigger.value == 'sell-bb_upper1':
            enter_short_conditions.append(df["close"] > df['&s-bb_upperband1'])
        if self.sell_trigger.value == 'sell-bb_upper2':
            enter_short_conditions.append(df["close"] > df['&s-bb_upperband2'])
        if self.sell_trigger.value == 'sell-bb_upper3':
            enter_short_conditions.append(df["close"] > df['&s-bb_upperband3'])
        if self.sell_trigger.value == 'sell-bb_upper4':
            enter_short_conditions.append(df["close"] > df['&s-bb_upperband4'])


        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < df[f"sell_roi_{self.std_dev_multiplier_sell.value}"] ,
            ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > df[f"target_roi_{self.std_dev_multiplier_buy.value}"] ,
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