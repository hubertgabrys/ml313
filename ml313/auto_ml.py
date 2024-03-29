import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from scipy.stats import kruskal
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


hyperparameter_space = {

    # Transformers
    'transformer:standard_scaler': {},
    'transformer:power_transformer': {},

    # Samplers
    'sampler:ros': {},
    'sampler:smote': {
        'k_neighbors': range(1, 11),
    },

    # Selectors
    'selector:remove_correlated': {
        'threshold': uniform(),
        'score_func': ['f-score', 'h-score'],
    },
    'selector:remove_nonnormal': {
        'skew_threshold': np.logspace(0, 2, 20, base=100),
        'kurt_threshold': np.logspace(0, 2, 20, base=100)
    },
    'selector:from_correlated2pca': {
        'n_components': 1.5 - np.logspace(-1, 0, 100, base=2),
    },
    'selector:sfm_lr': {
        'estimator__penalty': ['elasticnet'],
        'estimator__C': np.logspace(-4, 10, 1000, base=2),
        'estimator__l1_ratio': uniform(),
        'estimator__class_weight': [None, 'balanced'],
    },
    'selector:sfm_et': {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__criterion': ["gini", "entropy"],
        'estimator__max_features': np.arange(0.05, 1.01, 0.05),
        'estimator__min_samples_split': range(2, 21),
        'estimator__min_samples_leaf': range(1, 21),
        'estimator__bootstrap': [True, False],
        'estimator__class_weight': [None, 'balanced'],
    },
    'selector:sfm_gb': {
        'estimator__n_estimators': [100, 200, 300, 500],
        'estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'estimator__max_depth': range(1, 11),
        'estimator__min_samples_split': range(2, 21),
        'estimator__min_samples_leaf': range(1, 21),
        'estimator__subsample': np.arange(0.05, 1.01, 0.05),
        'estimator__max_features': np.arange(0.05, 1.01, 0.05)
    },
    'selector:sfm_xgb': {
        'estimator__n_estimators': [100, 200, 300, 500],
        'estimator__max_depth': range(1, 11),
        'estimator__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'estimator__subsample': np.arange(0.05, 1.01, 0.05),
        'estimator__min_child_weight': range(1, 21),
        'estimator__n_jobs': [1]
    },
    'selector:sfm_cgb': {
        'estimator__iterations': range(300, 3000),
        'estimator__depth': range(4, 12),
        'estimator__learning_rate': np.logspace(-2, -1, 1000, base=10),
        'estimator__random_strength': np.logspace(-9, 0, 1000, base=10),
        'estimator__bagging_temperature': uniform(),
        'estimator__border_count': range(1, 255),
        'estimator__l2_leaf_reg': range(2, 30),
        'estimator__scale_pos_weight': np.linspace(0.01, 1, 1000)
    },

    # Classifiers
    'classifier:lr': {
        'C': np.logspace(-5, 10, 100, base=2),
        'penalty': ['elasticnet'],
        'l1_ratio': uniform(),
        'class_weight': [None, 'balanced'],
    },
    'classifier:dt': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    'classifier:et': {
        'n_estimators': [100, 200, 300, 500],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced'],
    },
    'classifier:gb': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },
    'classifier:xgb': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1]
    },
    'classifier:cgb': {
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


class SelectNormal(BaseEstimator, SelectorMixin):
    def __init__(self, skew_threshold=None, kurt_threshold=None):
        self.skew_threshold = skew_threshold
        self.kurt_threshold = kurt_threshold
        self.support_ = None

    def fit(self, X, y=None):
        if self.skew_threshold is not None:
            X_skew = skew(X)
            skew_mask = np.abs(X_skew) < self.skew_threshold
        else:
            skew_mask = np.ones(X.shape[1], dtype=bool)
        if self.kurt_threshold is not None:
            X_kurt = kurtosis(X)
            kurt_mask = X_kurt < self.kurt_threshold
        else:
            kurt_mask = np.ones(X.shape[1], dtype=bool)
        self.support_ = np.logical_and(skew_mask, kurt_mask)
        return self

    def _get_support_mask(self):
        return self.support_


class SelectFromPCA(BaseEstimator, SelectorMixin):
    """Feature selection based on PCA.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.support_ = None

    def fit(self, X, y=None):
        # calculate pca
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X)

        # correlation between X and X_pca
        corr_mat = np.zeros((X.shape[1], X_pca.shape[1]))
        for feat_idx in range(X.shape[1]):
            for pca_idx in range(X_pca.shape[1]):
                r = pearsonr(X[:, feat_idx], X_pca[:, pca_idx])[0]
                corr_mat[feat_idx, pca_idx] = r

        # find most correlated column for each row
        corr_mat_abs = np.abs(corr_mat)
        corr_mat_max = np.max(corr_mat_abs, axis=1)
        corr_mat_masks = corr_mat_abs == corr_mat_max.reshape(-1, 1)

        # calculate f-scores
        f_scores = f_classif(X, y)[0]

        # calculate support
        support = np.zeros(X.shape[1], dtype=bool)
        for pca_idx in range(X_pca.shape[1]):
            if any(corr_mat_masks[:, pca_idx]):
                idx_max = np.argmax(corr_mat_masks[:, pca_idx] * f_scores)
                support[idx_max] = True
        self.support_ = support
        return self

    def _get_support_mask(self):
        return self.support_


class SelectKBestFromModel(BaseEstimator, SelectorMixin):
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
        self.mask_ = None

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

    def _get_support_mask(self):
        """Get a mask of the features selected."""
        try:
            scores = self.estimator.coef_[0, :]
        except (AttributeError, KeyError):
            scores = self.estimator.feature_importances_
        mask = np.zeros(len(scores))
        if self.k > len(scores):
            self.k = len(scores)
        mask[np.argpartition(abs(scores), -self.k)[-self.k:]] = 1
        self.mask_ = mask.astype(bool)
        return self.mask_


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

    def __init__(self, threshold=0.9, score_func='f-score'):
        self.threshold = threshold
        self.score_func = score_func

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
            X = pd.DataFrame(X)
            self.correlation_matrix_ = X.corr('pearson')
        # calculate the order of feature removal
        if self.score_func == 'f-score':
            F, pval = f_classif(X, y)
            index_arr = np.argsort(F)[::-1]
            self.order = X.columns[index_arr]
        elif self.score_func == 'h-score':
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
        'transformer:standard_scaler': StandardScaler(),
        'transformer:power_transformer': PowerTransformer(),
        'sampler:ros': RandomOverSampler(random_state=313),
        'sampler:smote': SMOTE(random_state=313),
        'selector:remove_correlated': CorrelationThreshold(),
        'selector:remove_nonnormal': SelectNormal(),
        'selector:from_correlated2pca': SelectFromPCA(),
        'selector:sfm_lr': SelectKBestFromModel(LogisticRegression(solver='saga', random_state=313)),
        'selector:sfm_et': SelectKBestFromModel(ExtraTreesClassifier(random_state=313)),
        'selector:sfm_gb': SelectKBestFromModel(GradientBoostingClassifier(random_state=313)),
        'selector:sfm_xgb': SelectKBestFromModel(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=313)),
        'classifier:lr': LogisticRegression(solver='saga', random_state=313),
        'classifier:dt': DecisionTreeClassifier(random_state=313),
        'classifier:et': ExtraTreesClassifier(random_state=313),
        'classifier:gb': GradientBoostingClassifier(random_state=313),
        'classifier:xgb': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=313),
    }
    steps = list()
    for step in template:
        steps.append((step, lookup_dict[step]))
    try:
        pipeline = Pipeline(steps=steps)
    except TypeError:
        pipeline = imbPipeline(steps=steps)
    return pipeline


def get_param_dist(pipeline, max_features=None):
    param_dist = {}
    for step in pipeline.steps:
        step_params = hyperparameter_space[step[0]]
        step_params = {step[0] + f'__{k}': v for k, v in step_params.items()}
        if step[0][:3] == 'sfm':
            step_params[step[0] + '__k'] = np.arange(1, max_features + 1)
        param_dist = {**param_dist, **step_params}
    return param_dist


def get_support(model):
    no_feats = None
    i = 0
    while not no_feats:
        try:
            no_feats = model.best_estimator_.steps[i][1].get_support().shape[0]
        except AttributeError:
            i += 1
    indices = np.arange(no_feats)
    for step in model.best_estimator_.steps:
        if isinstance(step[1], SelectorMixin) or isinstance(step[1], SelectKBestFromModel):
            indices = indices[step[1].get_support()]
    support = np.zeros(no_feats, dtype=bool)
    support[indices] = True
    return support


def get_weights(model):
    try:
        weights = model.best_estimator_.steps[-1][1].coef_[0]
    except AttributeError:
        weights = model.best_estimator_.steps[-1][1].feature_importances_
    return weights
