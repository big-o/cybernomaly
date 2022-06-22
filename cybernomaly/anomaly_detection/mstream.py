import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from cybernomaly.anomaly_detection.base import Monitor


class MStream(Monitor):
    def __init__(self, dimensionality_reduction=None, thresh=None, batch_size=None):
        self.dimensionality_reduction = dimensionality_reduction or IncrementalPCA()
        try:
            if not callable(self.dimensionality_reduction.fit):
                raise TypeError("fit attribute must be callable")
            if not callable(self.dimensionality_reduction.transform):
                raise TypeError("transform attribute must be callable")
        except (TypeError, AttributeError):
            raise ValueError(
                "dimensionality_reduction parameter must be a valid sklearn "
                f"transformer. Got '{dimensionality_reduction}' instead."
            )

        self.batch_size = batch_size
        self.thresh = thresh

    def _check_X_y(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        X, y = self._validate_data(
            X, y,
            reset=first_pass,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        return X, y

    def partial_fit(self, X, y):
        X, y = self._check_X_y(X, y)
        first_pass = self._check_partial_fit_first_call()
        if first_pass:
            self._init_setup(X, y)

        self._extend_fit_buffer(X, y)

        if self._has_partial_fit and self._get_fit_buffer_size() == self.batch_size_:
            self.dimensionality_reduction.partial_fit(self._X_fit, self._y_fit)
            self._reset_fit_buffer()


        return self

    def _init_setup(self, X, y):
        self._reset_fit_buffer()
        n_samples, n_features = X.shape
        self.thresh_ = self.thresh
        self.batch_size_ = self.batch_size if self.batch_size is not None else 5 * n_features
        self._has_partial_fit = hasattr(self.dimensionality_reduction, "partial_fit")
        self._reset_fit_buffer()

    def _reset_fit_buffer(self):
        self._X_fit = None
        self._y_fit = None

    def _get_fit_buffer_size(self):
        return len(self._X_fit)

    def _extend_fit_buffer(self, X, y):
        self._X_fit = np.vstack((self._X_fit, X)) if self._X_fit else np.copy(X)
        self._y_fit = np.vstack((self._y_fit, y)) if self._X_fit else np.copy(X)

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self._init_setup(X, y)
        self.dimensionality_reduction.fit(X, y)
        return self

    def detect_score(self, X):
        if self._get_fit_buffer_size() > 0:
            if self._has_partial_fit:
                self.dimensionality_reduction.partial_fit(self._X_fit, self._y_fit)
            else:
                self.dimensionality_reduction.fit(self._X_fit, self._y_fit)
            self._reset_fit_buffer()

        Xtr = self.dimensionality_reduction.transform(X)
        score = self._mstream(Xtr)
        return score

    def _mstream(self, X):
        pass

    def detect(self, X):
        return self.detect_score(X) > self.thresh_

    def update(self, X):
        pass

    def update_detect(self, X):
        self.update(X)
        return self.detect(X)

    def update_detect_score(self, X):
        self.update(X)
        return self.detect_score(X)

    def _check_partial_fit_first_call(self):
        return hasattr(self, "_X_fit")
