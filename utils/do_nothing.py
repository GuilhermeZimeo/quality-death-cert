"""DoNothing class which bypass transformers."""
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin

class DoNothing(BaseEstimator, TransformerMixin):
    """Do nothing and return the estimator unchanged.

    Parameters
    ----------
    X : array-like

    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        check_array(X, accept_sparse='csr')
        return self

    def transform(self, X, copy=None):
        X = check_array(X, accept_sparse='csr')
        return X

    def fit_transform(self, X, y=None, copy=None):
        X = check_array(X, accept_sparse='csr')
        return X
