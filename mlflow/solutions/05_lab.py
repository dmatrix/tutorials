import keras
import mlflow
from sol_cls.keras_model import MNistKerasModel

"""
Aim of this module is:
1. Introduce tracking ML experiments in MLflow
2. Record parameters, metrics, and a model
"""

# TODO in LAB
#    Add layers
#    Make hidden units larger
#    Try a different optimizer: RMSprop and Adadelta
#    Train for more epochs
#    change these default parameters are and observe how it will effect the results

if __name__ == '__main__':
    params_list = [
            {'batch_size': 128,
              'epochs': 5,
              'learning_rate': 0.05,
              'num_inputs_units': 512,
              'num_hidden_layers': 1,
              'dropout': 0.24,
              'momentum': 0.85,
              'optimizer': 'SGD'},
              {'batch_size': 128,
               'epochs': 10,
               'learning_rate': 0.05,
               'num_inputs_units': 512,
               'num_hidden_layers': 2,
               'dropout': 0.24,
               'momentum': 0.85,
               'optimizer': 'RMSprop'},
              {'batch_size': 128,
               'epochs': 20,
               'learning_rate': 0.05,
               'num_inputs_units': 512,
               'num_hidden_layers': 4,
               'dropout': 0.24,
               'momentum': 0.85,
               'optimizer': 'RMSprop'}
            ]

    # get MNIST data set
    mnist = keras.datasets.mnist
    # normalize the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Set local registry store
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # create model
    for params in params_list:
        tfkm = MNistKerasModel.new_instance(x_train, y_train, x_test, y_test, params)
        print(tfkm)
        # Track the model runs
        tfkm.mlfow_run()


