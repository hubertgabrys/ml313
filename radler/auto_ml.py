import numpy as np
from scipy.stats import uniform
import pandas as pd
from scipy.stats import kruskal
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

hyperparameter_space = {
    'sfpk': {},
    'sfpk_meanct': {},
    'sfpk_msskct': {},
    'sfpk_meanpet': {},
    'sfpk_msskpet': {},
    'sfpk_volume': {},
    'sfpk_shape': {},
    'sfpk_intensity': {},
    'sfpk_texture': {},
    'sfpk_wavelets': {},
    'standard_scaler': {},
    'power_transformer': {},
    'samp_ros': {},
    'samp_smote': {
        'k_neighbors': range(1,11),
        },
    'decorr': {
        'threshold': uniform(),
        'order': ['natural', 'f-score', 'h-score'],
        },
    'sfm_lr': {
        'estimator__penalty': ['l1', 'l2', 'elasticnet'],
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
        'estimator__nthread': [1]
        }, 
    'clf_lr': {
        'C': np.logspace(-5, 10, 1000, base=2),
        'penalty': ['l1', 'l2', 'elasticnet'],
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
        'nthread': [1]
        }, 
    }


class SelectFromPriorKnowledge(BaseEstimator, SelectorMixin):
    """Feature selection based on prior knowledge.
    If used, it needs to be the first step in the pipeline.
    """

    def __init__(self, selected_features=None):
        self.selected_features = selected_features
        self.all_features = None

    def fit(self, X, y=None):
        self.all_features = X.columns.values
        return self

    def _get_support_mask(self):
        return np.in1d(self.all_features, self.selected_features, assume_unique=True)


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
        'sfpk': SelectFromPriorKnowledge(),
        'sfpk_volume': SelectFromPriorKnowledge(selected_features=['CT0_mc-volume']),
        'sfpk_meanct': SelectFromPriorKnowledge(selected_features=['CT0_mean']),
        'sfpk_msskct': SelectFromPriorKnowledge(selected_features=['CT0_mean', 'CT0_sd', 'CT0_skewness', 'CT0_kurtosis']),
        'sfpk_meanpet': SelectFromPriorKnowledge(selected_features=['PET0_mean']),
        'sfpk_msskpet': SelectFromPriorKnowledge(selected_features=['PET0_mean', 'PET0_sd', 'PET0_skewness', 'PET0_kurtosis']),
        'standard_scaler': StandardScaler(),
        'power_transformer': PowerTransformer(),
        'samp_ros': RandomOverSampler(random_state=313),
        'samp_smote': SMOTE(random_state=313),
        'decorr': CorrelationThreshold(),
        'sfm_lr': SelectFromModel(LogisticRegression(solver='saga', random_state=313), threshold=-np.inf),
        'sfm_et': SelectFromModel(ExtraTreesClassifier(random_state=313)),
        'sfm_gb': SelectFromModel(GradientBoostingClassifier(random_state=313)),
        'sfm_xgb': SelectFromModel(XGBClassifier(random_state=313)),
        'clf_lr': LogisticRegression(solver='saga', random_state=313),
        'clf_dt': DecisionTreeClassifier(random_state=313),
        'clf_et': ExtraTreesClassifier(random_state=313),
        'clf_gb': GradientBoostingClassifier(random_state=313),
        'clf_xgb': XGBClassifier(random_state=313)
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
