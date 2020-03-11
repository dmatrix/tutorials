"""

Problem - part 1: We want to predict the gas consumption in millions of gallons in 48 of the US states
based on some key features. These features are petrol tax (in cents), per capital income (in US dollars),
paved highway (in miles), population of people with driving licences

Solution:

Since this is a regression problem where the value is a range of numbers, we can use the
common Random Forest Algorithm in Scikit-Learn. Most regression models are evaluated with
three standard evalution metrics: Mean Absolute Error(MAE); Mean Squared Error (MSE); and
Root Mean Squared Error (RSME), and r2.

This example is borrowed from the source below, modified and modularized for this tutorial
source: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

Aim of this Lab:

1. Understand MLflow Tracking API
2. How to use the MLflow Tracking API
3. Use the MLflow API to experiment several Runs
4. Interpret and observer runs via the MLflow UI

Some Resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://www.saedsayad.com/decision_tree_reg.htm
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""

import mlflow.sklearn
from lab_cls.rfr_model import RFRModel
from lab_cls.lab_utils import Utils
#
# TODO in Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestRegressor documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators.
    # 4. Check in MLflow UI if the metrics are affected
    # challenge-1: create root mean square error and r2 artifacts and save them for each run

if __name__ == '__main__':
    # Use sqlite:///mlruns.db as the local store for tracking and registery
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # load and print dataset
    dataset = Utils.load_data("data/petrol_consumption.csv")
    Utils.print_pandas_dataset(dataset)
    # iterate over several runs with different parameters,
    for n in range (25, 125, 25):
    # stepping up by 25 trees and limiting to 100
        params = {"n_estimators": n, "random_state": 0 }
        rfr = RFRModel.new_instance(params)
        (experimentID, runID) = rfr.mlflow_run(dataset)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)
