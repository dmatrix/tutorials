"""

Problem - part 2: Given a set of features or attributes of a bank note, can we predict whether it's authentic or fake
Four attributes contribute to this classification:
1. variance or transformed image
2. skeweness
3. entropy
4. cutoosis

Solution:

We are going to use Random Forest Classification to make the prediction, and measure on the accuracy.
The closer to 1.0 is the accuracy the better is our confidence in its prediction.

This example is borrowed from the source below, modified and modularized for this tutorial
source: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
source:https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Aim of this Lab:

1. Understand MLflow Tracking API
2. How to use the MLflow Tracking API
3. Use the MLflow API to experiment several Runs
4. Interpret and observer runs via the MLflow UI

Some resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://devopedia.org/confusion-matrix
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
"""

import mlflow.sklearn
from lab_cls.rfc_model import RFCModel
from lab_cls.lab_utils import Utils

#
# Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestClassifier documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators
    # 4. Check in MLfow UI if the metrics are affected
    # 5. Log confusion matirx, recall and F1-score as metrics
    # Nice blog: https://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/

if __name__ == '__main__':
    # load and print dataset
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    dataset = Utils.load_data("data/bill_authentication.csv")
    Utils.print_pandas_dataset(dataset)
    # iterate over several runs with different parameters
    # TODO in the Lab (change these parameters, n_estimators and random_state
    # with each iteration.
    # Does that change the metrics and accuracy?
    # start with n=10, step by 10 up to X <=100
    for n in range(10, 30, 10):
        params = {"n_estimators": n, "random_state": 0 }
        rfr = RFCModel.new_instance(params)
        (experimentID, runID) = rfr.mlflow_run(dataset)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)

