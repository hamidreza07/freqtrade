# Trade by models:
1. Sample Model and Config running:
```bash
freqtrade trade --strategy PivotPoint --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel CatboostClassifierMultiTarget --freqaimodel-path freqtrade/freqai/prediction_models/ 
```

2. SVM:
* Don't use SVM.I've tested before ; RMSE was very high.

3. Pytorch:
* DO NOT CREATE ANOTHER CLASS OR FILE AT  freqai/torch PASS. IT'S ONLY ACCEPT ONE FILE AS A MODEL WITH THE EXACT NAME FOR CLASS AND FILE __PyTorchMLPModel__

## backtesting:
1. download the data:
```bash
freqtrade download-data --exchange binance   --timerange 20230520-20230815 --timeframes 1w 5m 1d   --trading-mode futures --config config_examples/config_freqai.example.json
```
2. run backtesting:
```bash
freqtrade backtesting -s PivotPoint --strategy-path freqtrade/templates  --config config_examples/config_freqai.example.json   --timerange 20230601-20230701
```

## hyperparameter optimazer:
1. download data:
```bash
freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230801-20230904 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures
```

2. run optimazer:
```bash
freqtrade hyperopt -s PivotPoint --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --hyperopt-loss OnlyProfitHyperOptLoss -e 40 --timerange 20230601-20230801
```

## plot the backtest:

1. download data:
```bash
freqtrade download-data --exchange binance --timeframes 1w  --timerange 20230601-20230701    --trading-mode futures --config config_examples/config_freqai.example.json
```
2. Run thic command:
```bash
freqtrade plot-dataframe --strategy PivotPoint --strategy-path freqtrade/templates   --userdir  user_data/  -c  config_examples/config_freqai.example.json  --timerange 20230601-20230701 --freqaimodel-path freqtrade/freqai/prediction_models
```





* be carefull of train_period_days and backtest_period_days in running backtest and hyperparameter optimazer at freqai(sum of train_period_days and backtest_period_days must not be greater than time range)

* Remove all - __pycache__ - file before backtesting and hyperopt:

```bash
find . -type d -name "__pycache__" -exec rm -r {} \; ; rm -rf user_data/* ; rm -rf trad*

```
