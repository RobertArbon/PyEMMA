from __future__ import absolute_import

"""
Reshapes trajectories so that they can be stored as a single array of [X[i],X[i+tau]] observations.
This makes CV much more robust as folds are no longer along whole trajectories but are now
along single observations.
Downside is that you must specify a single tau and keep with it for the whole pipeline.
"""

"""
This is a module to be used as a reference for building other modules
"""


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class DtrajReshape(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y)


        # Return the estimator
        return self

    def transform(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        X = check_array(X)
        X_list = [X[i,:].T[:, np.newaxis] for i in range(X.shape[0])]
        return X_list

