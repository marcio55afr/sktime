# -*- coding: utf-8 -*-
"""Debug distances."""

import time

import numpy as np
import pandas as pd
from tslearn.metrics import dtw as tslearn_dtw

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datatypes import convert_to
from sktime.distances import distance_factory
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.tests._utils import create_test_distance_numpy

# dataset_name = "Beef"
# dataset_name = "ACSF1"
dataset_name = "PenDigits"
X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)
knn = KNeighborsTimeSeriesClassifier(distance="euclidean")
knn2 = KNeighborsTimeSeriesClassifier(distance="dtw")


def _run_experiment(distance, name=None):
    """Run experiment."""
    print("++++++++++++++++")
    if name is None:
        name = distance

    knn = KNeighborsTimeSeriesClassifier(distance=distance)
    start = int(round(time.time() * 1000))
    knn.fit(X_train, y_train)
    build_time = int(round(time.time() * 1000)) - start
    print(name + " = ", knn.score(X_test, y_test))
    predict_time = int(round(time.time() * 1000)) - start
    print(
        name + " fit time = ", build_time / 1000, " total time = ", predict_time / 1000
    )


if __name__ == "__main__":
    _run_experiment("dtw")
    _run_experiment(dtw_distance, "cython dtw")
    _run_experiment(tslearn_dtw, "tslearn dtw")
    print("done")
