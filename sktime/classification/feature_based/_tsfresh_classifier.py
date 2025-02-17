# -*- coding: utf-8 -*-
"""TSFresh Classifier.

Pipeline classifier using the TSFresh transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["TSFreshClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)


class TSFreshClassifier(BaseClassifier):
    """Time Series Feature Extraction based on Scalable Hypothesis Tests classifier.

    This classifier simply transforms the input data using the TSFresh [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    relevant_feature_extractor : bool, default=False
        Remove irrelevant features using the FRESH algorithm.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to a
        Random Forest with 200 trees.
    verbose : int, default=0
        level of output printed to the console (for information only)
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.

    See Also
    --------
    TSFreshFeatureExtractor, TSFreshRelevantFeatureExtractor

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh–a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Examples
    --------
    >>> from sktime.classification.feature_based import TSFreshClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = TSFreshClassifier(
    ...     default_fc_parameters="minimal",
    ...     estimator=RandomForestClassifier(n_estimators=10),
    ... )
    >>> clf.fit(X_train, y_train)
    TSFreshClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        default_fc_parameters="efficient",
        relevant_feature_extractor=False,
        estimator=None,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.relevant_feature_extractor = relevant_feature_extractor
        self.estimator = estimator

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self._transformer = None
        self._estimator = None

        super(TSFreshClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = (
            TSFreshRelevantFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._threads_to_use,
                chunksize=self.chunksize,
            )
            if self.relevant_feature_extractor
            else TSFreshFeatureExtractor(
                default_fc_parameters=self.default_fc_parameters,
                n_jobs=self._threads_to_use,
                chunksize=self.chunksize,
            )
        )
        self._estimator = _clone_estimator(
            RandomForestClassifier(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        if self.verbose < 2:
            self._transformer.show_warnings = False
            if self.verbose < 1:
                self._transformer.disable_progressbar = True

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._threads_to_use

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self._class_dictionary[preds[i]]] = 1
            return dists
