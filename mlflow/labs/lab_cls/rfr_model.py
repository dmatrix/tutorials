import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from . lab_utils import Utils

class RFRModel():
    """
    General class for Scikit-Learn RandomForestRegressor
    """
    # class wide variables common to all instances
    # keep track of cumulative estimators and rsme
    # so we can plot the results
    rsme = []
    estimators = []

    def __init__(self, params={}):
        """
        Constructor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self.rf = RandomForestRegressor(**params)
        self._params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Return the model craeted
        :return: handle or instance of the RandomForestReqgressor
        """
        return self.rf

    @property
    def params(self):
        """
        Getter for model params
        """
        return self._params

    def mlflow_run(self, df, r_name="Lab-1:RF Petrol Regression Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the run as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all feature independent attributes
            X = df.iloc[:, 0:4].values
            # get all the values of last columns, dependent variables,
            # which is what we want to predict as our values, the petrol consumption
            y = df.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling, though for RF is not necessary.
            # z = (X - u)/ s, where u is the man, s the standard deviation
            # get the handle to the transformer
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.model, "random-forest-reg-model")
            mlflow.log_params(self.params)

            # compute  metrics; r2 is a statistical measure of how well the
            # data fits the model: higher the value indicates better fit.
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            # update global class instance variable with values
            self.rsme.append(rsme)
            self.estimators.append(self._params["n_estimators"])

            # plot graphs and save as artifacts
            (fig, ax) = Utils.plot_graphs(self.estimators, self.rsme, "Random Forest Estimators", "Root Mean Square", "Root Mean Square vs Estimators")

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("rsme_estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "rsme_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)

            return (experimentID, runID)
