import pandas as pd
import numpy as np
import time

from sktime.datasets import load_UCR_UEA_dataset
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.distances import distance_factory
from sktime.datatypes import convert_to
from sktime.distances.elastic_cython import dtw_distance

dataset_name = "Beef"
# dataset_name = "ACSF1"
X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train",
return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test",
return_X_y=True)
knn = KNeighborsTimeSeriesClassifier(distance="euclidean")
knn2 = KNeighborsTimeSeriesClassifier(distance="dtw")

def run_experiment(distance, name=None):
    print("++++++++++++++++")
    if name is None:
        name = distance

    if name == 'distance factory':
        convert_x = convert_to(X_train, 'numpy3D')
        convert_x = convert_x.transpose((0, 2, 1))
        distance = distance_factory(convert_x[0], convert_x[1], metric=distance)

    knn = KNeighborsTimeSeriesClassifier(distance=distance)
    start = int(round(time.time() * 1000))
    knn.fit(X_train, y_train)
    build_time = int(round(time.time() * 1000)) - start
    print(name + " = ", knn.score(X_test, y_test))
    predict_time = int(round(time.time() * 1000)) - start
    print(name + " fit time = ", build_time / 1000, " total time = ", predict_time / 1000)


if __name__ == '__main__':
    run_experiment('dtw')
    run_experiment(dtw_distance, "cython dtw")
    print("done")
