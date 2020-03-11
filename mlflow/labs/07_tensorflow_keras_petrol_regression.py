import tensorflow as tf
import mlflow

from lab_cls.lab_utils import Utils
from lab_cls.tf_keras_model import TFKerasRegModel

if __name__ =='__main__':
    print("Using TensorFlow Version={}".format(tf.__version__))
    params_list = [
        {'input_units': 64,
              'input_shape': (4,),
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'epochs': 100,
              'batch_size': 128},
        {'input_units': 128,
              'input_shape': (4,),
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'epochs': 200,
              'batch_size': 128},
        {'input_units': 256,
            'input_shape': (4,),
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mse',
            'epochs': 200,
            'batch_size': 128}
        ]

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    dataset = Utils.load_data("data/petrol_consumption.csv")
    # get all feature independent attributes
    X = dataset.iloc[:, 0:4].values
    # get all the values of last columns, dependent variables,
    # which is what we want to predict as our values, the petrol consumption
    y = dataset.iloc[:, 4].values
    for params in params_list:
        keras_model = TFKerasRegModel(params)
        (experimentID, runID) = keras_model.mlflow_run(X, y)
        print("MLflow completed with run_id {} and experiment_id {}".format(runID, experimentID))
