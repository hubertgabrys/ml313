from imblearn.base import BaseSampler
from sklearn.ensemble import IsolationForest


class IForest(BaseSampler):
    """iForest removes outliers from the data.
    Parameters
    ----------
    threshold : float, optional
        All observations below the threshold will be removed
        The default threshold is 0.0.
    n_estimators : int, optional
        The number of base estimators in the ensemble.
        The default value is 100.
    Attributes
    ----------
    isf_scores : array, shape (n_features,n_features)
        Scores from isolation forest.
    """

    def __init__(self, threshold=0.0, n_estimators=1000):
        super().__init__()
        self._sampling_type = "bypass"
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.isf = IsolationForest(n_estimators=self.n_estimators)
        self.isf_scores_ = None

    def _fit_resample(self, X, y):
        self.isf.fit(X)
        self.isf_scores_ = self.isf.decision_function(X)
        isf_support = self.isf_scores_ > self.threshold
        return X[isf_support], y[isf_support]