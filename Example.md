# Trade by models:
1. arima
- freqtrade trade --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/arima.json --freqaimodel ARIMAModel --freqaimodel-path freqtrade/freqai/prediction_models/ 

2. KNeighborsRegressorModel
- freqtrade trade --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/KNeighborsRegressor.json --freqaimodel KNeighborsRegressorModel --freqaimodel-path freqtrade/freqai/prediction_models/ 

## backtesting:
1. download the data:
- freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230101-20230901 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures

2. run backtesting:
- freqtrade backtesting --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel CatboostRegressor  --timerange 20230101-20230901 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --freqaimodel-path freqtrade/freqai/prediction_models/ 


## hyperparameter optimazer:
1. download data:
- freqtrade download-data --exchange binance --timeframes 3m  --timerange 20230101-20230901 --pairs BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT  --erase --trading-mode futures
2. run optimazer:

- freqtrade hyperopt -s Strategy005 --strategy-path freqtrade/templates  --freqaimodel ARIMAModel --freqaimodel-path freqtrade/freqai/prediction_models --config config_examples/config_freqai.example.json --hyperopt-loss OnlyProfitHyperOptLoss -p BTC/USDT:USDT ETH/USDT:USDT XRP/USDT:USDT -e 400

