"""Custom imputers for the golf model pipeline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with a specified percentile (e.g., 25th).
    
    Players without PGA Tour stats (Korn Ferry fill-ins, amateurs) are
    imputed with the 25th percentile instead of the median, reflecting
    that they're likely worse than the average tour player.
    """
    def __init__(self, percentile=25):
        self.percentile = percentile

    def fit(self, X, y=None):
        self.fill_values_ = np.nanpercentile(X, self.percentile, axis=0)
        return self

    def transform(self, X):
        X = np.copy(X)
        mask = np.isnan(X)
        X[mask] = np.take(self.fill_values_, np.where(mask)[1])
        return X
