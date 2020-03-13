import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from . lab_utils import Utils

class RFCModel():

    def __init__(self, params={}):
        """
        Constructor for RandamForestClassifier
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self.rf = RandomForestClassifier(**params)
        self.params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def model(self):
        """
        Fetch the model
        :return: return the model
        """
        return self.rf

    def mlflow_run(self, df, r_name="Solution-2:RF Bank Note Classification Experiment"):

        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all rows and columns but the last column, which is our class
            X = df.iloc[:, 0:4].values
            # get all observed values in the last columns, which is what we want to predict
            y = df.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.rf, "random-forest-class-model")
            mlflow.log_params(self.params)

            # compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)

            # get confusion matrix values
            true_positive = conf_matrix[0][0]
            true_negative = conf_matrix[1][1]
            false_positive = conf_matrix[0][1]
            false_negative = conf_matrix[1][0]

            # get confusion matrix as a dictionary
            class_report = classification_report(y_test,y_pred, output_dict=True)
            recall_0 = class_report['0']['recall']
            f1_score_0 = class_report['0']['f1-score']
            recall_1 = class_report['1']['recall']
            f1_score_1 = class_report['1']['f1-score']

            # log metrics
            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("true_positive", true_positive)
            mlflow.log_metric("true_negative", true_negative)
            mlflow.log_metric("false_positive", false_positive)
            mlflow.log_metric("false_negative", false_negative)
            mlflow.log_metric("recall_0", recall_0)
            mlflow.log_metric("f1_score_0", f1_score_0)
            mlflow.log_metric("recall_1", recall_1)
            mlflow.log_metric("f1_score_1", f1_score_1)

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create confusion matrix images
            (plt, fig, ax) = Utils.plot_confusion_matrix(y_test, y_pred, y, title="Bank Note Classification Confusion Matrix")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("confusion_matrix-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "confusion_matrix_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimators trees:", self.params["n_estimators"])
            print(conf_matrix)
            print(classification_report(y_test,y_pred))
            print("Accuracy Score:", acc)
            print("Precision     :", precision)

            return (experimentID, runID)
