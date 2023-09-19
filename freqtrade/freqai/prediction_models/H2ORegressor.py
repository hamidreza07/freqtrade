import h2o
from h2o.estimators import H2OGradientBoostingEstimator

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

class H2ORegressor(BaseRegressionModel):
    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> any:
        h2o.init()  # Initialize H2O

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None
        else:
            test_data = h2o.H2OFrame(data_dictionary["test_features"].join(data_dictionary["test_labels"]))
            eval_set = test_data

        train_data = h2o.H2OFrame(data_dictionary["train_features"].join(data_dictionary["train_labels"]))
        

        model = H2OGradientBoostingEstimator(**self.model_training_parameters)

        model.train(x=train_data.columns[:-1], y=train_data.columns[-1], 
                    training_frame=train_data, 
                    validation_frame=eval_set,
                  )
        
    
        return model
