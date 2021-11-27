# -*- coding: utf-8 -*-
"""CIT vector classifier.

Continuous Interval Tree aka Time Series Tree (TST), base classifier originally used
in the TimeSeriesForest interval based classification algorithm.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ContinuousIntervalTree"]

import math
import sys

import numpy as np
import scipy.stats
from numba import njit
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_X_y

from sktime.exceptions import NotFittedError
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X


class ContinuousIntervalTree(BaseEstimator):
    """Continuous Interval Tree (CIT).

    The 'Time Series Tree' described in the Time Series Forest (TSF) paper Deng et al
    (2013). [1]_
    A simple information gain based tree for continuous attributes using a bespoke
    margin gain metric for tie breaking.
    Implemented for interval based time series classifiers such as
    CanonicalIntervalForest and DrCIF.

    Parameters
    ----------
    max_depth : int, default=sys.maxsize
        Maximum depth for the tree.
    thresholds : int, default=20
        Number of thresholds to split attributes on at tree nodes.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    root : _TreeNode
        Tree root node.

    See Also
    --------
    CanonicalIntervalForest, DrCIF

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    machine_learning/classifiers/ContinuousIntervalTree.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction",Information Sciences, 239, 2013

    Examples
    --------
    >>> from sktime.contrib.vector_classifiers._continuous_interval_tree import ContinuousIntervalTree
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> X_train = from_nested_to_3d_numpy(X_train)
    >>> X_test = from_nested_to_3d_numpy(X_test)
    >>> clf = ContinuousIntervalTree()
    >>> clf.fit(X_train, y_train)
    ContinuousIntervalTree(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        max_depth=sys.maxsize,
        thresholds=20,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.thresholds = thresholds

        self.random_state = random_state

        # The following set in method fit
        self.root = None
        self.n_classes = 0
        self.classes_ = []

        self._class_dictionary = {}
        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

        super(ContinuousIntervalTree, self).__init__()

    def fit(self, X, y):
        """Fit a tree on cases (X,y), where y is the target variable.

        Build an information gain based tree for continuous attributes using the
        margin gain metric for ties.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,n_attributes]
        The training input samples.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A 2d numpy array is required."
            )
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        rng = check_random_state(self.random_state)
        self.root = _TreeNode(random_state=rng)

        thresholds = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.thresholds)
        distribution_cls, distribution = unique_count(y)
        e = _entropy(distribution, distribution.sum())

        self.root.build_tree(
            X,
            y,
            thresholds,
            e,
            distribution_cls,
            distribution,
            0,
            self.max_depth,
            False,
        )

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances,n_attributes]

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances,n_attributes]

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A 2d numpy array is required."
            )

        dists = np.zeros((X.shape[0], self.n_classes))
        for i in range(X.shape[0]):
            dists[i] = self.root.predict_proba(X[i], self.n_classes)
        return dists

    def _predict_proba_cif(self, X, c22, intervals, dims, atts):
        """Embedded predict proba for the CIF classifier."""
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        X = check_X(X, coerce_to_numpy=True)
        n_instances, n_dims, series_length = X.shape

        dists = np.zeros((n_instances, self.n_classes))
        for i in range(n_instances):
            dists[i] = self.root.predict_proba_cif(
                X[i].reshape((1, n_dims, series_length)),
                c22,
                intervals,
                dims,
                atts,
                self.n_classes,
            )
        return dists

    def _predict_proba_drcif(
        self, X, X_p, X_d, c22, n_intervals, intervals, dims, atts
    ):
        """Embedded predict proba for the DrCIF classifier."""
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        X = check_X(X, coerce_to_numpy=True)
        n_instances, n_dims, series_length = X.shape

        dists = np.zeros((n_instances, self.n_classes))
        for i in range(n_instances):
            r = [
                X[i].reshape((1, n_dims, series_length)),
                X_p[i].reshape((1, n_dims, X_p.shape[2])),
                X_d[i].reshape((1, n_dims, X_d.shape[2])),
            ]
            dists[i] = self.root.predict_proba_drcif(
                r,
                c22,
                n_intervals,
                intervals,
                dims,
                atts,
                self.n_classes,
            )
        return dists

    def tree_node_splits_and_gain(self):
        """Recursively find the split and information gain for each tree node."""
        splits = []
        gains = []

        if self.root.best_split > -1:
            self._find_splits_gain(self.root, splits, gains)

        return splits, gains

    def _find_splits_gain(self, node, splits, gains):
        """Recursively find the split and information gain for each tree node."""
        splits.append(node.best_split)
        gains.append(node.best_gain)

        for next_node in node.children:
            if next_node.best_split > -1:
                self._find_splits_gain(next_node, splits, gains)


class _TreeNode:
    """ContinuousIntervalTree tree node."""

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

        self.best_split = -1
        self.best_threshold = 0
        self.best_gain = 0.000001
        self.best_margin = -1
        self.children = []
        self.leaf_distribution_cls = []
        self.leaf_distribution = []
        self.depth = -1

    def build_tree(
        self,
        X,
        y,
        thresholds,
        entropy,
        distribution_cls,
        distribution,
        depth,
        max_depth,
        leaf,
    ):
        self.depth = depth
        best_splits = []
        best_distributions_cls = []
        best_distributions = []
        best_entropies = []

        if leaf is False and depth < max_depth:
            for (_, att), threshold in np.ndenumerate(thresholds):
                (
                    info_gain,
                    splits,
                    distributions_cls,
                    distributions,
                    entropies,
                ) = self.information_gain(X, y, att, threshold, entropy)

                if info_gain > self.best_gain:
                    self.best_split = att
                    self.best_threshold = threshold
                    self.best_gain = info_gain
                    self.best_margin = -1
                    best_splits = splits
                    best_distributions_cls = distributions_cls
                    best_distributions = distributions
                    best_entropies = entropies
                elif info_gain == self.best_gain and info_gain > 0.000001:
                    margin = self.margin_gain(X, att, threshold)
                    if self.best_margin == -1:
                        self.best_margin = self.margin_gain(
                            X, self.best_split, self.best_threshold
                        )

                    if margin > self.best_margin or (
                        margin == self.best_margin
                        and self.random_state.choice([True, False])
                    ):
                        self.best_split = att
                        self.best_threshold = threshold
                        self.best_margin = margin
                        best_splits = splits
                        best_distributions_cls = distributions_cls
                        best_distributions = distributions
                        best_entropies = entropies

        if self.best_split > -1:
            self.children = [None, None, None]

            if sum(best_splits[0]) > 0:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X[best_splits[0]],
                    y[best_splits[0]],
                    thresholds,
                    best_entropies[0],
                    best_distributions_cls[0],
                    best_distributions[0],
                    depth + 1,
                    max_depth,
                    len(best_distributions[0]) == 1,
                )
            else:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution_cls,
                    distribution,
                    depth + 1,
                    max_depth,
                    True,
                )

            if sum(best_splits[1]) > 0:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X[best_splits[1]],
                    y[best_splits[1]],
                    thresholds,
                    best_entropies[1],
                    best_distributions_cls[1],
                    best_distributions[1],
                    depth + 1,
                    max_depth,
                    len(best_distributions[1]) == 1,
                )
            else:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution_cls,
                    distribution,
                    depth + 1,
                    max_depth,
                    True,
                )

            if sum(best_splits[2]) > 0:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X[best_splits[2]],
                    y[best_splits[2]],
                    thresholds,
                    best_entropies[2],
                    best_distributions_cls[2],
                    best_distributions[2],
                    depth + 1,
                    max_depth,
                    len(best_distributions[2]) == 1,
                )
            else:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution_cls,
                    distribution,
                    depth + 1,
                    max_depth,
                    True,
                )
        else:
            self.leaf_distribution_cls = list(distribution_cls)
            self.leaf_distribution = list(distribution / distribution.sum())

        return self

    def predict_proba(self, X, n_classes):
        if self.best_split > -1:
            if np.isnan(X[self.best_split]):
                return self.children[0].predict_proba(X, n_classes)
            elif X[self.best_split] <= self.best_threshold:
                return self.children[1].predict_proba(X, n_classes)
            else:
                return self.children[2].predict_proba(X, n_classes)
        else:
            dist = np.zeros(n_classes)
            for i, prob in enumerate(self.leaf_distribution):
                dist[int(self.leaf_distribution_cls[i])] = prob
            return dist

    def predict_proba_cif(self, X, c22, intervals, dims, atts, n_classes):
        if self.best_split > -1:
            interval = int(self.best_split / len(atts))
            att = self.best_split % len(atts)
            value = _cif_feature(X, intervals[interval], dims[interval], atts[att], c22)
            value = value.round(8)
            value = np.nan_to_num(value, False, posinf=np.nan, neginf=np.nan)

            if np.isnan(value):
                return self.children[0].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            elif value <= self.best_threshold:
                return self.children[1].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            else:
                return self.children[2].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
        else:
            dist = np.zeros(n_classes)
            for i, prob in enumerate(self.leaf_distribution):
                dist[int(self.leaf_distribution_cls[i])] = prob
            return dist

    def predict_proba_drcif(
        self,
        X,
        c22,
        n_intervals,
        intervals,
        dims,
        atts,
        n_classes,
    ):
        if self.best_split > -1:
            rep = -1
            rep_sum = 0
            for i in range(len(X)):
                rep_sum += n_intervals[i] * len(atts)
                if self.best_split < rep_sum:
                    rep = i
                    break

            interval = int(self.best_split / len(atts))
            att = self.best_split % len(atts)

            value = _drcif_feature(
                X[rep], intervals[interval], dims[interval], atts[att], c22
            )
            value = value.round(8)
            value = np.nan_to_num(value, False, posinf=np.nan, neginf=np.nan)

            # currently unused
            if np.isnan(value):
                return self.children[0].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            elif value <= self.best_threshold:
                return self.children[1].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            else:
                return self.children[2].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
        else:
            dist = np.zeros(n_classes)
            for i, prob in enumerate(self.leaf_distribution):
                dist[int(self.leaf_distribution_cls[i])] = prob
            return dist

    @staticmethod
    @njit(fastmath=True, cache=True)
    def information_gain(X, y, attribute, threshold, parent_entropy):
        missing = np.isnan(X[:, attribute])
        dist_missing_cls, dist_missing = unique_count(y[missing])
        left = X[:, attribute] <= threshold
        dist_left_cls, dist_left = unique_count(y[left])
        right = X[:, attribute] > threshold
        dist_right_cls, dist_right = unique_count(y[right])

        sum_missing = 0
        for v in dist_missing:
            sum_missing += v
        sum_left = 0
        for v in dist_left:
            sum_left += v
        sum_right = 0
        for v in dist_right:
            sum_right += v

        entropy_missing = _entropy(dist_missing, sum_missing)
        entropy_left = _entropy(dist_left, sum_left)
        entropy_right = _entropy(dist_right, sum_right)

        num_cases = X.shape[0]
        info_gain = (
            parent_entropy
            - sum_missing / num_cases * entropy_missing
            - sum_left / num_cases * entropy_left
            - sum_right / num_cases * entropy_right
        )

        return (
            info_gain,
            [missing, left, right],
            [dist_missing_cls, dist_left_cls, dist_right_cls],
            [dist_missing, dist_left, dist_right],
            [entropy_missing, entropy_left, entropy_right],
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def margin_gain(X, attribute, threshold):
        margins = np.abs(X[:, attribute] - threshold)
        return np.min(margins)


@njit(fastmath=True, cache=True)
def unique_count(x):
    if len(x) > 0:
        x = np.sort(x)
        unique = np.zeros(len(x))
        unique[0] = x[0]
        counts = np.zeros(len(x), dtype=np.int_)
        counts[0] = 1
        unique_count = 0

        for i in x[1:]:
            if i != unique[unique_count]:
                unique_count += 1
                unique[unique_count] = i
                counts[unique_count] = 1
            else:
                counts[unique_count] += 1
        return unique[: unique_count + 1], counts[: unique_count + 1]
    return None, np.zeros(0, dtype=np.int_)


@njit(fastmath=True, cache=True)
def _entropy(x, s):
    e = 0
    for i in x:
        p = i / s if s > 0 else 0
        e += -(p * math.log(p) / 0.6931471805599453) if p > 0 else 0
    return e


def _cif_feature(X, interval, dim, att, c22):
    if att == 22:
        # mean
        return np.mean(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 23:
        # std_dev
        return np.std(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 24:
        # slope
        return _slope(X[:, dim, interval[0] : interval[1]], axis=1)
    else:
        return c22._transform_single_feature(
            X[:, dim, interval[0] : interval[1]],
            feature=att,
        )


def _drcif_feature(X, interval, dim, att, c22):
    if att == 22:
        # mean
        return np.mean(X[:, dim, interval[0] : interval[1]], axis=1)
    if att == 23:
        # median
        return np.median(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 24:
        # std_dev
        return np.std(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 25:
        # slope
        return _slope(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 26:
        # iqr
        return scipy.stats.iqr(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 27:
        # min
        return np.min(X[:, dim, interval[0] : interval[1]], axis=1)
    elif att == 28:
        # max
        return np.max(X[:, dim, interval[0] : interval[1]], axis=1)
    else:
        return c22._transform_single_feature(
            X[:, dim, interval[0] : interval[1]],
            feature=att,
        )
