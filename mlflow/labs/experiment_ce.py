import os
import shutil

from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":

    # set the tracking server to be Databricks Community Edition
    # set the experiment name; if name does not exist, MLflow will
    # create one for you
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/jules.damji@gmail.com/Jules_CE_Test")
    print("Running experiment_ce.py")
    print("Tracking on https://community.cloud.databricks.com")

    mlflow.start_run(run_name="CE_TEST")
    params = {"n_estimators": 3, "random_state": 0}
    rfr = RandomForestRegressor(params)

    # Log model and params using the MLflow sklearn APIs
    mlflow.sklearn.log_model(rfr, "random-forest-reg-model")
    mlflow.log_params(params)
    log_param("param_1", randint(0, 100))

    log_metric("metric_1", random())
    log_metric("metric_2", random() + 1)
    log_metric("metric_3", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("Looks, like I logged on the Community Edition!")

    log_artifacts("outputs")
    shutil.rmtree('outputs')
    mlflow.end_run()



