import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from . lab_utils import Utils

class RFRBaseModel():

    def __init__(self, params={}):
        """
        Construtor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self.params = params
        self.rf = RandomForestRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def model(self):
        """
        Getter for the model
        :return: return the model
        """
        return self.rf

    def mlflow_run(self, df, r_name="Lab-3: Baseline RF Model"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: Tuple of MLflow experimentID, runID
        """
        with mlflow.start_run(run_name=r_name) as run:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.rf, "random-forest-model")
            mlflow.log_params(self.params)

            # Create metrics
            mae = metrics.mean_absolute_error(y_test, predictions)
            mse = metrics.mean_squared_error(y_test, predictions)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # print some data
            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)

            return (experimentID, runID)


class RFRExperimentModel(RFRBaseModel):

    """
    Constructor for the Experimental RandomForestRegressor.
    """

    def __init__(self, params):

        """
        Call the superclass initializer
        :param params: parameters for the RandomForestRegressor instance
        :return: None
        """
        super(RFRExperimentModel, self).__init__(params)

    def mlflow_run(self, df, experiment, r_name="Lab-4:RF Experiment Model"):
        """
        Override the base class mlflow_run for this epxerimental runs
        This method trains the model, evaluates, computes the metrics, logs
        all the relevant metrics, artifacts, and models.
        :param df: pandas dataFrame
        :param r_name: name of the experiment run
        :return:  MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1),
                                                                df[["price"]].values.ravel(), random_state=42)
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.rf, "random-forest-model")

            # Note we are logging as a dictionary of all params instead of logging each parameter
            mlflow.log_params(self.params)

            # Log params
            # [mlflow.log_param(param, value) for param, value in self.params.items()]

            # Create metrics
            mse = metrics.mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(y_test, predictions)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rsme", rmse)
            mlflow.log_metric("r2", r2)

            # get experimentalID and runID
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # Create feature importance and save them as artifact
            # This allows us to remove least important features from the dataset
            # with each iteration if they don't have any effect on the predictive power of
            # the prediction.
            importance = pd.DataFrame(list(zip(df.columns, self.rf.feature_importances_)),
                                      columns=["Feature", "Importance"]
                                      ).sort_values("Importance", ascending=False)

            # Log importance file as feature artifact
            temp_file_name = Utils.get_temporary_directory_path("feature-importance-", ".csv")
            temp_name = temp_file_name.name
            try:
                importance.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "feature-importance-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create residual plots and image directory
            # Residuals R = observed value - predicted value
            (plt, fig, ax) = Utils.plot_residual_graphs(predictions, y_test, "Predicted values for Price ($)",
                                                        "Residual", "Residual Plot")

            # Log residuals images
            temp_file_name = Utils.get_temporary_directory_path("residuals-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "residuals-plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(experiment, runID, experimentID))

            print("  mse: {}".format(mse))
            print(" rmse: {}".format(rmse))
            print("  mae: {}".format(mae))
            print("  R2 : {}".format(r2))

            return (experimentID, runID)
