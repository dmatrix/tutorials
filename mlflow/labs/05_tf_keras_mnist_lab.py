import tensorflow as tf
import keras

import mlflow
import mlflow.tensorflow

# Callback function for logging metrics at at of each epoch
class LogMetricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        mlflow.log_metric("training_loss", logs["loss"], epoch)
        mlflow.log_metric("training_accuracy", logs["accuracy"], epoch)

class MNistKerasModel:

    def __init__(self, X_train, Y_train, X_test, Y_test, params={}):
        self._params = params
        self._X_train = X_train
        self._Y_train = Y_train
        self._X_test = X_test
        self._Y_test = Y_test

    def __repr__(self):
        return f"TF/Keras model built with parameters = {self.params}"

    @classmethod
    def new_instance(cls, X_train, Y_train, X_test, Y_test, params={}):

        return cls(X_train, Y_train, X_test, Y_test, params)

    @property
    def params(self):
        """
        Getter for model parameters
        """
        return self._params

    @property
    def params(self, params):
        self._params = params

    @params.getter
    def params(self):
        return self._params

    @params.setter
    def params(self, key, value):
        self._params[key] = value

    def model(self):
        model_attr = "tf_keras_mnist_model"
        model_reg = getattr(self, model_attr, None)
        if model_reg is None:
            try:
                model_reg = self._build_compiled_model()
                setattr(self, model_attr, model_reg)
            except Exception as e:
                raise Exception(e)
        return model_reg

    def _build_compiled_model(self):
        # build the model
        tf_model = tf.keras.models.Sequential()
        # The first layer in this network transforms the format of the images from a 2d-array (of 28 by 28 pixels),
        # to a 1d-array of 28 * 28 = 784 pixels.
        # The first layer in this network transforms the format of the images from a 2d-array (of 28 by 28 pixels),
        # to a 1d-array of 28 * 28 = 784 pixels.
        tf_model.add(tf.keras.layers.Flatten(input_shape=self._X_train[0].shape)),
        # add extra hidden layers to expand the NN
        # --num_hidden_layers or -N  in the command line arguments
        for n in range(0, self.params['num_hidden_layers']):
            tf_model.add(tf.keras.layers.Dense(self.params['num_inputs_units'], activation='relu')),
        # dropout is a regularization technique for NN where we randomly dropout a layer if the
        # computed gradients are minimal or have no effect.
        tf_model.add(tf.keras.layers.Dropout(self.params['dropout'])),
        # final layer with softmax activation layer
        tf_model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # Use Scholastic Gradient Descent (SGD)
        # https://keras.io/optimizers/
        #
        optimizer = self.get_optimizer(self.params['optimizer'])

        # compile the model with optimizer and loss type
        # common loss types for classification are
        # 1. sparse_categorical_crossentropy
        # 2. binary_crossentropy
        tf_model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return tf_model

    def get_optimizer(self, opt_name):
        """
        :param name: name of the Keras optimizer
        :param args: args for the optimizer
        :return: Keras optimizer
        """

        if opt_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(lr=self.params['learning_rate'],
                                             momentum=self.params['momentum'],
                                             nesterov=True)
        elif (opt_name == 'RMSprop'):
            optimizer = tf.keras.optimizers.RMSprop(lr=self.params['learning_rate'], rho=0.9, epsilon=None, decay=0.0)
        else:
            optimizer = tf.keras.optimizers.Adadelta(lr=self.params['learning_rate'], epsilon=None, decay=0.0)

        return optimizer

    def mlfow_run(self, run_name="Lab-5:TensorFlow/Keras_MNIST"):
        """
        Method to run MLflow experiment
        :return: Tuple (experiment_id, run_id)
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Callback function for logging metrics at at of each epoch
            # fit the model
            # get experiment id and run id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            print("-" * 100)
            print("MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            self.model().fit(self._X_train, self._Y_train,
                      epochs=self.params['epochs'],
                      batch_size=self.params['batch_size'],
                      callbacks=[LogMetricsCallback()])
            # evaluate the model
            test_loss, test_acc = self.model().evaluate(self._X_test, self._Y_test, verbose=2)
            # log metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)

            # log model as native Keras Model
            mlflow.keras.log_model(self.model(), artifact_path="tf-keras-mnist-model")

            # get experiment id and run id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            return (experimentID, runID)


if __name__ =='__main__':

    print("Using TensorFlow Version={}".format(tf.__version__))
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
