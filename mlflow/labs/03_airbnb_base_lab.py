"""
source: Databricks Learning Academy Lab

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
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://github.com/MangoTheCat/Modelling-Airbnb-Prices
"""

import mlflow.sklearn
from lab_cls.rfr_base_exp_model import RFRBaseModel
from lab_cls.lab_utils import Utils

#
# TODO in Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForest documentation
    # 2. Run the baseline model
    # 3. Check in MLflow UI for parameters, metrics, and artifacts

if __name__ == '__main__':
    # load and print dataset
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    dataset = Utils.load_data("data/airbnb-cleaned-mlflow.csv")
    Utils.print_pandas_dataset(dataset)
    #
    # create a base line model parameters
    # this is our benchmark model to compare experimental results with
    #
    params = {"n_estimators": 100, "max_depth": 3, "random_state": 0}
    rfr = RFRBaseModel.new_instance(params)
    (experimentID, runID) = rfr.mlflow_run(dataset)
    print("MLflow completed with run_id {} and experiment_id {}".format(runID, experimentID))
    print("-" * 100)
