# Trade by models:
1. arima
- freqtrade trade --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/arima.json --freqaimodel ARIMAModel --freqaimodel-path freqtrade/freqai/prediction_models/ 

2. KNeighborsRegressorModel
- freqtrade trade --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/KNeighborsRegressor.json --freqaimodel KNeighborsRegressorModel --freqaimodel-path freqtrade/freqai/prediction_models/ 

## backtesting:
1. download the data:
- freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230101-20230901 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures

2. run backtesting:
- freqtrade backtesting -s FreqaiExampleStrategy --strategy-path freqtrade/templates  --freqaimodel ARIMAModel --freqaimodel-path freqtrade/freqai/prediction_models --config config_examples/arima_config.json  -p BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --timerange 20230510-20230610


## hyperparameter optimazer:
1. download data:
- freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230801-20230904 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures
2. run optimazer:

-  freqtrade hyperopt -s FreqaiExampleStrategy --strategy-path freqtrade/templates  --freqaimodel ARIMAModel --freqaimodel-path freqtrade/freqai/prediction_models --config config_examples/arima_config.json --hyperopt-loss OnlyProfitHyperOptLoss -p BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT -e 400 --timerange 20230825-20230905


* be carefull of train_period_days and backtest_period_days in running backtest and hyperparameter optimazer at freqai(sum of train_period_days and backtest_period_days must not be greater than time range)