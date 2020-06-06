import numpy as np
from scipy.stats import uniform
import pandas as pd
from scipy.stats import kruskal
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


hyperparameter_space = {
    'standard_scaler': {},
    'power_transformer': {},
    'samp_ros': {},
    'samp_smote': {
        'k_neighbors': range(1,11),
        },
    'decorr': {
        'threshold': uniform(),
        'order': ['f-score', 'h-score'],
        },
    'sfm_lr': {
        'estimator__penalty': ['elasticnet'],
        'estimator__C': np.logspace(-4, 10, 1000, base=2),
        'estimator__l1_ratio': uniform(),
        'estimator__class_weight': [None, 'balanced'],
        },
    'sfm_et': {
        'estimator__n_estimators': [100, 200, 300, 500],
        'estimator__criterion': ["gini", "entropy"],
        'estimator__max_features': np.arange(0.05, 1.01, 0.05),
        'estimator__min_samples_split': range(2, 21),
        'estimator__min_samples_leaf': range(1, 21),
        'estimator__bootstrap': [True, False],
        'estimator__class_weight': [None, 'balanced'],
        },
    'sfm_gb': {
        'estimator__n_estimators': [100, 200, 300, 500],
        'estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'estimator__max_depth': range(1, 11),
        'estimator__min_samples_split': range(2, 21),
        'estimator__min_samples_leaf': range(1, 21),
        'estimator__subsample': np.arange(0.05, 1.01, 0.05),
        'estimator__max_features': np.arange(0.05, 1.01, 0.05)
        },
    'sfm_xgb': {
        'estimator__n_estimators': [100, 200, 300, 500],
        'estimator__max_depth': range(1, 11),
        'estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'estimator__subsample': np.arange(0.05, 1.01, 0.05),
        'estimator__min_child_weight': range(1, 21),
        'estimator__n_jobs': [1]
        },
    'sfm_cgb': {
        'estimator__iterations': range(300, 3000),
        'estimator__depth': range(4, 12),
        'estimator__learning_rate': np.logspace(-2, -1, 1000, base=10),
        'estimator__random_strength': np.logspace(-9, 0, 1000, base=10),
        'estimator__bagging_temperature': uniform(),
        'estimator__border_count': range(1, 255),
        'estimator__l2_leaf_reg': range(2, 30),
        'estimator__scale_pos_weight': np.linspace(0.01, 1, 1000)
        },
    'clf_lr': {
        'C': np.logspace(-5, 10, 1000, base=2),
        'penalty': ['elasticnet'],
        'l1_ratio': uniform(),
        'class_weight': [None, 'balanced'],
        },
    'clf_dt': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    'clf_et': {
        'n_estimators': [100, 200, 300, 500],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced'],
        },
    'clf_gb': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
        },
    'clf_xgb': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1]
        },
    'clf_cgb': {
        'iterations': range(500, 2000),
        'depth': range(4, 12),
        'learning_rate': np.logspace(-2, -1, 1000, base=10),
        'random_strength': np.logspace(-9, 0, 1000, base=10),
        'bagging_temperature': uniform(),
        'border_count': range(1, 255),
        'l2_leaf_reg': range(2, 30),
        'scale_pos_weight': np.linspace(0.01, 1, 1000)
        },
    }


class SelectKBestFromModel(BaseEstimator, TransformerMixin):
    """Feature selection based on k-best features of a fitted model.
    It corresponds to a recursive feature elimination with a single step.
    """

    def __init__(self, estimator, k=3):
        """Initialize the object.
        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a ``fit`` method that provides
            information about feature importance either through a ``coef_``
            attribute or through a ``feature_importances_`` attribute.
        k : int, default=3
            The number of features to select.
        """
        self.estimator = estimator
        self.k = k
        self.mask = None

    def fit(self, X, y=None):
        """Fit the underlying estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        self.estimator.fit(X, y)
        return self

    def get_support(self):
        """Get a mask of the features selected."""
        try:
            scores = self.estimator.coef_[0, :]
        except (AttributeError, KeyError):
            scores = self.estimator.feature_importances_
        mask = np.zeros(len(scores))
        if self.k > len(scores):
            self.k = len(scores)
        mask[np.argpartition(abs(scores), -self.k)[-self.k:]] = 1
        self.mask = mask.astype(bool)

    def transform(self, X):
        """Reduce X to the selected features.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        self.get_support()
        return X[:, self.mask]


class CorrelationThreshold(BaseEstimator, SelectorMixin):
    """Feature selector that removes all highly-correlated features.
    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.
    Parameters
    ----------
    threshold : float, optional
        Features correlated higher than this threshold will be removed.
        The default threshold is 0.90.
    Attributes
    ----------
    correlation_matrix_ : array, shape (n_features,n_features)
        Correlation matrix.
    """

    def __init__(self, threshold=0.9, order='natural'):
        self.threshold = threshold
        self.order = order
        self.correlation_matrix_ = None

    def fit(self, X, y=None):
        """Learn empirical variances from X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self
        """
        # calculate correlation matrix
        if isinstance(X, pd.DataFrame):
            self.correlation_matrix_ = X.corr('pearson')
        else:
            self.correlation_matrix_ = pd.DataFrame(X).corr('pearson')
        # calculate the order of feature removal
        if self.order == 'natural':
            self.order = X.columns
        elif self.order == 'f-score':
            F, pval = f_classif(X, y)
            index_arr = np.argsort(F)[::-1]
            self.order = X.columns[index_arr]
        elif self.order == 'h-score':
            h_stat = list()
            for col in X.columns:
                statistic, pvalue = kruskal(X.loc[y, col], X.loc[~y, col])
                h_stat.append(statistic)
            h_stat = np.asarray(h_stat)
            index_arr = np.argsort(h_stat)[::-1]
            self.order = X.columns[index_arr]
        return self

    def _get_support_mask(self):
        for col in self.order:
            if col in self.correlation_matrix_.index:
                mask = np.abs(self.correlation_matrix_[col]) < self.threshold
                mask[col] = True
                self.correlation_matrix_ = self.correlation_matrix_.loc[mask, :]
        return np.array([e in self.correlation_matrix_.index for e in self.correlation_matrix_.columns])


def get_pipeline(template):
    lookup_dict = {
        'standard_scaler': StandardScaler(),
        'power_transformer': PowerTransformer(),
        'samp_ros': RandomOverSampler(random_state=313),
        'samp_smote': SMOTE(random_state=313),
        'decorr': CorrelationThreshold(),
        'sfm_lr': SelectKBestFromModel(LogisticRegression(solver='saga', random_state=313)),
        'sfm_et': SelectKBestFromModel(ExtraTreesClassifier(random_state=313)),
        'sfm_gb': SelectKBestFromModel(GradientBoostingClassifier(random_state=313)),
        'sfm_xgb': SelectKBestFromModel(XGBClassifier(random_state=313)),
        'sfm_cgb': SelectKBestFromModel(CatBoostClassifier(random_state=313, eval_metric='AUC', verbose=0)),
        'clf_lr': LogisticRegression(solver='saga', random_state=313),
        'clf_dt': DecisionTreeClassifier(random_state=313),
        'clf_et': ExtraTreesClassifier(random_state=313),
        'clf_gb': GradientBoostingClassifier(random_state=313),
        'clf_xgb': XGBClassifier(random_state=313),
        'clf_cgb': CatBoostClassifier(random_state=313, eval_metric='AUC', verbose=0),
        }
    steps = list()
    for step in template:
        steps.append((step, lookup_dict[step]))
    try:
        pipeline = Pipeline(steps=steps)
    except TypeError:
        pipeline = imbPipeline(steps=steps)
    return pipeline 


def get_param_dist(pipeline, max_features=None, sel_features=None):
    param_dist = {}
    for step in pipeline.steps:
        step_params = hyperparameter_space[step[0]]
        step_params = {step[0] + f'__{k}': v for k, v in step_params.items()}
        if step[0][:3] == 'sfm':
            step_params[step[0] + '__max_features'] = np.arange(1, max_features + 1)
        if step[0] == 'sfpk':
            step_params[step[0] + '__selected_features'] = [sel_features]
        param_dist = {**param_dist, **step_params}
    return param_dist


def get_support(model):
    no_feats = model.best_estimator_.steps[0][1].get_support().shape[0]
    indices = np.arange(no_feats)
    for step in model.best_estimator_.steps:
        if isinstance(step[1], SelectorMixin):
            indices = indices[step[1].get_support()]
    support = np.zeros(no_feats, dtype=bool)
    support[indices] = True
    return support


def get_weights(model):
    try:
        weights = model.best_estimator_.steps[-1][1].coef_
    except AttributeError:
        weights = model.best_estimator_.steps[-1][1].feature_importances_
    return weights
