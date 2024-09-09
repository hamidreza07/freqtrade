# Trade by models:
1. Sample Model and Config running:
```bash
freqtrade trade --strategy EmaEngAI --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel GridRegressionDTMultiTarget --freqaimodel-path freqtrade/freqai/prediction_models/ 
```

2. SVM:
* Don't use SVM.I've tested before ; RMSE was very high.

3. Pytorch:
* DO NOT CREATE ANOTHER CLASS OR FILE AT  freqai/torch PASS. IT'S ONLY ACCEPT ONE FILE AS A MODEL WITH THE EXACT NAME FOR CLASS AND FILE __PyTorchMLPModel__

## backtesting:
1. download the data:
```bash
freqtrade download-data --exchange binance   --timerange 20230420-20230820 --timeframes  5m    --trading-mode futures --config config_examples/config_freqai.example.json
```
2. run backtesting:
```bash
freqtrade backtesting -s EmaEngAI --strategy-path freqtrade/templates  --config config_examples/config_freqai.example.json   --timerange 20230601-20230701 --freqaimodel GridRegressionDTMultiTarget --freqaimodel-path freqtrade/freqai/prediction_models/ 
```

## hyperparameter optimazer:
1. download data:
```bash
freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230801-20230904 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures
```

2. run optimazer:
```bash
freqtrade hyperopt -s EmaEngAI --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --hyperopt-loss OnlyProfitHyperOptLoss -e 40 --timerange 20230501-20230601 --freqaimodel GridRegressionDTMultiTarget --freqaimodel-path freqtrade/freqai/prediction_models/ 
```

## plot the backtest:

1. download data:
```bash
freqtrade download-data --exchange binance --timeframes 1w  --timerange 20230601-20230701    --trading-mode futures --config config_examples/config_freqai.example.json
```
2. Run thic command:
```bash
freqtrade plot-dataframe --strategy BBvwap --strategy-path freqtrade/templates   --userdir  user_data/  -c  config_examples/config_freqai.example.json  --timerange 20230601-20230701 --pairs BTC/USDT:USDT 
```





* Be mindful of the `train_period_days` and `backtest_period_days` settings when running backtests and hyperparameter optimizations in FreqAI. The sum of `train_period_days` and `backtest_period_days` must not exceed the specified time range.

* Remove all - __pycache__ - file before backtesting and hyperopt:

```bash
find . -type d -name "__pycache__" -exec rm -r {} \; ; rm -rf user_data/* ; rm -rf trad*

```
