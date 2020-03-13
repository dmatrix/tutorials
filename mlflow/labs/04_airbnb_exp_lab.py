"""
Databricks Learning Academy Lab -

Refactored code to modularize it

While iterating or build models, data scientists will often create a base line model to see how the model performs.
And then iterate with experiments, changing or altering parameters to ascertain how the new parameters or
hyper-parameters move the metrics closer to their confidence level.

This is our base line model using RandomForestRegressor model to predict AirBnb house prices in SF.
Given 22 features can we predict what the next house price will be?

We will compute standard evalution metrics and log them.

Aim of this module is:

1. Introduce tracking ML experiments in MLflow
2. Log a base experiment and explore the results in the UI
3. Record parameters, metrics, and a model

Some Resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://www.saedsayad.com/decision_tree_reg.htm
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
https://seaborn.pydata.org/tutorial/regression.html
"""

import mlflow.sklearn
from  mlflow.tracking import MlflowClient
from lab_cls.rfr_base_exp_model import RFRExperimentModel

from lab_cls.lab_utils import Utils

# Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestRegressor documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators
    # 4. Check in MLfow UI if the metrics are affected

if __name__ == '__main__':
    # TODO add more parameters to the list
    # create four experiments with different parameters
    # run these different experiments, each with its own instance of model with the supplied parameters.
    # add more parameters to this dictionary list here
    params_list = [
        {"n_estimators": 200,"max_depth": 6, "random_state": 42}
    ]
    # load the data
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    dataset = Utils.load_data("data/airbnb-cleaned-mlflow.csv")

    # run these experiments, each with its own instance of model with the supplied parameters.
    for params in params_list:
        rfr = RFRExperimentModel.new_instance(params)
        experiment = "Experiment with {} trees".format(params['n_estimators'])
        (experimentID, runID) = rfr.mlflow_run(dataset, experiment)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)

    # Use MLflowClient API to query programmatically any previous run info under an experiment ID
    # consult https://mlflow.org/docs/latest/python_api/mlflow.tracking.html
    client = MlflowClient()
    run_list = client.list_run_infos(experimentID)
    [print(rinfo) for rinfo in run_list]

