import mlflow
import keras
import tensorflow as tf
from lab_cls.keras_model import MNistKerasModel

if __name__ =='__main__':

    print("Using TensorFlow Version={}".format(tf.__version__))
    # Set local registry store
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # Our model parameters that can be tweaked for regularization and
    #
    params = {'batch_size': 128,
              'epochs': 5,
              'learning_rate': 0.05,
              'num_inputs_units': 512,
              'num_hidden_layers': 1,
              'dropout': 0.24,
              'momentum':0.85,
              'optimizer': 'SGD'}
    # get MNIST data set
    mnist = keras.datasets.mnist
    # normalize the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # create model
    tfkm = MNistKerasModel.new_instance(x_train, y_train, x_test, y_test, params)
    print(tfkm)
    # track the model runs
    tfkm.mlfow_run()
