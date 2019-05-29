"""Classes and functions for multivariate models."""


import copy
import itertools as it
import pdb
import random
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import scikits.bootstrap as bootstrap
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import (AllKNN, ClusterCentroids,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     InstanceHardnessThreshold, NearMiss,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours,
                                     TomekLinks)
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import (RFE, SelectKBest, f_classif,
                                       mutual_info_classif, VarianceThreshold)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (BaseCrossValidator, RandomizedSearchCV,
                                     StratifiedShuffleSplit, cross_val_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from skrebate import SURF, MultiSURF, MultiSURFstar, ReliefF, SURFstar
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from .univariate import mann_whitney_u, recursive_reduction



class InlierDetection(BaseEstimator):
    """Usupervised inlier detection."""

    def __init__(self, method='none'):
        self.method = method


    def fit_resample(self, X, y):
        r"""Fit the statistics and resample the data directly.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in x.

        Returns
        -------
        X_resampled : {array-like, sparse matrix}, shape \
        (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : array-like, shape (n_samples_new,)
            The corresponding label of 'x_resampled'.

        """
        if self.method == 'none':
            return X, y
        elif self.method == 'isolation_forest':
            self.selector = IsolationForest()
            mask = self.selector.fit(X).predict(X) == 1
            X_reduced = X[mask]
            y_reduced = y[mask]
            return X_reduced, y_reduced


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

    def __init__(self, threshold=0.9, n_jobs=1):
        self.threshold = threshold
        self.n_jobs = n_jobs

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
        self.correlation_matrix_ = X.corr('pearson')

        return self

    def _get_support_mask(self):
        corrmat = self.correlation_matrix_
        order = corrmat.columns
        for col in order:
            if col in corrmat.index:
                mask = np.abs(corrmat[col]) < self.threshold
                mask[col] = True
                corrmat = corrmat[mask]
        big_mask = np.array([e in corrmat.index for e in corrmat.columns])

        return big_mask


class MWWFeatureSelection(BaseEstimator, SelectorMixin):
    """Description.
    Parameters
    ----------
    corr_threshold : float, optional
        Features correlated higher than this threshold will be removed.
        The default threshold is 0.90.
    Attributes
    ----------
    correlation_matrix_ : array, shape (n_features,n_features)
        Correlation matrix.
    """

    def __init__(self, corr_method='pearson', corr_threshold=0.9, n_feats=-1, n_jobs=1):
        self.corr_method = corr_method
        self.corr_threshold = corr_threshold
        self.n_feats = n_feats
        self.n_jobs = n_jobs

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
        self.df_auc_ = mann_whitney_u(X, y, boot_iters=1000, n_jobs=self.n_jobs, verbose=0)
        self.df_corr_ = X.corr(self.corr_method)
        self.reduced_feats_ = recursive_reduction(self.df_auc_, self.df_corr_, threshold=self.corr_threshold, retain=None)

        return self

    def _get_support_mask(self):
        n_feats = int(np.round(len(self.reduced_feats_) * self.n_feats))
        if n_feats < 2:
            n_feats = 2
        sel_feats = self.reduced_feats_[:n_feats]
        mask = np.array([e in sel_feats for e in self.df_auc_.index])
        return mask


class Sampling(BaseEstimator):
    """A class handling sampling for class-balancing and noise removal.

    A documentation and references for the used methods:
    http://contrib.scikit-learn.org/imbalanced-learn/stable/user_guide.html
    """

    sampling_methods = ['cc', 'cnn', 'enn', 'renn', 'aknn', 'iht', 'nm', 'ncr',
                        'oss', 'rus', 'tl', 'adasyn', 'ros', 'smote',
                        'smoteenn', 'smotetomek']

    def __init__(self, kind, cc, cnn, enn, renn, aknn, iht, nm, ncr, oss, rus,
                 tl, adasyn, ros, smote, smoteenn, smotetomek):
        """Initialize the object.

        Parameters
        ----------
        kind : str, {'none', 'cc', 'cnn', 'enn', 'renn', 'aknn', 'iht', 'nm',
            'ncr', 'oss', 'rus', 'tl', 'adasyn', 'ros', 'smote', 'smoteenn',
            'smotetomek'}.
            Specifies the method used for sampling.
        cc : bool
            Cluster centroids.
        cnn : bool
            Condensed nearest neighbour.
        enn : bool
            Edited nearest neighbours.
        renn : bool
            Repeated edited nearest neighbours.
        aknn : bool
            All k-NN.
        iht : bool
            Instance hardness threshold.
        nm : bool
            Near miss.
        ncr : bool
            Neighbourhood-cleaning rule.
        oss : bool
            One-sided selection.
        rus : bool
            Random undersampling.
        tl : bool
            TomekLinks.
        adasyn : bool
            ADASYN.
        ros : bool
            Random oversampling.
        smote : bool
            SMOTE.
        smoteenn : bool
            SMOTE followed by ENN.
        smotetomek : bool
            SMOTE followed by Tomek links.

        """
        self.kind = kind
        for key in self.sampling_methods:
            exec('self.' + key + ' = ' + key)

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        if self.kind == 'none':
            return self
        else:
            return eval('self.' + self.kind + '.fit(X, y)')

    def fit_resample(self, X, y):
        r"""Fit the statistics and resample the data directly.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in x.

        Returns
        -------
        X_resampled : {array-like, sparse matrix}, shape \
        (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : array-like, shape (n_samples_new,)
            The corresponding label of 'x_resampled'.

        """
        if self.kind == 'none':
            return X, y
        else:
            return eval('self.' + self.kind + '.fit_sample(X, y)')

    def sample(self, X, y):
        r"""Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in x.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
        (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of 'x_resampled;.

        """
        if self.kind == 'none':
            return X, y
        else:
            return eval('self.' + self.kind + '.sample(X, y)')


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
        except AttributeError:
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
        X_r = X[:, self.mask]
        return X_r


class FeatureSelection(BaseEstimator, TransformerMixin):
    """A class handing supervised feature selection.

    A documentation for the used methods:
    1. http://scikit-learn.org/stable/modules/feature_selection.html
    2. https://epistasislab.github.io/scikit-rebate/
    3. https://en.wikipedia.org/wiki/Relief_(feature_selection)
    """

    feature_selection_methods = ['ufs_anova', 'ufs_mi', 'rfe_lr', 'rfe_et',
                                 'mb_lr', 'mb_et', 'mb_xgb', 'relieff', 'surf', 'surfs',
                                 'msurf', 'msurfs']

    def __init__(self, kind, k, ufs_anova, ufs_mi, rfe_lr, rfe_et, mb_lr,
                 mb_et, mb_xgb, relieff, surf, surfs, msurf, msurfs):
        """Initialize the object.

        Parameters
        ----------
        kind : str, {'none', 'ufs_anova', 'ufs_mi', 'rfe_lr', 'rfe_et',
                     'mb_lr', 'mb_et', 'mb_xgb', 'relieff', 'surf', 'surfs', 'msurf',
                     'msurfs'}.
            Specifies the method used for feature selection.
        k : int
            Number of features to select.
        ufs_anova : bool
            Feature selection based on ANOVA F-value.
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
        ufs_mi : bool
            Feature selection based on mutual information.
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
        rfe_lr : bool
            Recursive feature elimination based on logistic regression.
        rfe_et : bool
            Recursive feature elimination based on extra-trees.
        mb_lr : bool
            Model-based feature selection based on logistic regression.
        mb_et : bool
            Model-based feature selection based on extra-trees.
        mb_xgb : bool
            Model-based feature selection based on gradient tree boosting.
        relieff : bool
            ReliefF feature selection.
        surf : bool
            SURF feature selection.
        surfs : bool
            SURF* feature selection.
        msurf : bool
            MultiSURF feature selection.
        msurfs : bool
            MultiSURF* feature selection.

        """
        # print('fs_init')
        self.kind = kind
        for key in self.feature_selection_methods:
            exec('self.' + key + ' = ' + key)
            exec('self.' + key + '.k = k')
            exec('self.' + key + '.n_features_to_select = k')

    def fit(self, X, y=None):
        """Run the featue selection method to get the appropriate features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        """
        # Reduce number of features to select if needed
        if self.k <= X.shape[1]:
            for key in self.feature_selection_methods:
                exec('self.' + key + '.k = self.k')
                exec('self.' + key + '.n_features_to_select = self.k')
        else:
            k = X.shape[1]
            for key in self.feature_selection_methods:
                exec('self.' + key + '.k = ' + str(k))
                exec('self.' + key + '.n_features_to_select = ' + str(k))
        # Return the right method
        if self.kind == 'none':
            return self
        else:
            return eval('self.' + self.kind + '.fit(X, y)')

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
        if self.kind == 'none':
            return X
        else:
            return eval('self.' + self.kind + '.transform(X)')


class Classification(BaseEstimator, ClassifierMixin):
    """A class handing classification.

    A documentation for the used methods:
    1. http://scikit-learn.org/stable/supervised_learning.html
    2. https://xgboost.readthedocs.io/en/latest/
    """

    classification_methods = ['lr1', 'lr2', 'sgd', 'knn', 'svm', 'et', 'xgb']

    def __init__(self, kind, lr1, lr2, sgd, knn, svm, et, xgb):
        """Initialize the object.

        Parameters
        ----------
        kind : str, {'lr1', 'lr2', 'sgd', 'knn', 'svm', 'et', 'xgb'}.
            Specifies the method used for classification.
        lr1 : bool
            Logistic regression with L1 penalty.
        lr2 : bool
            Logistic regression with L2 penalty.
        sgd : bool
            Logistic regression with elastic-net penalty.
        knn : bool
            k-nearest neighbors.
        svm : bool
            Support vector machine with radial basis function (RBF) kernel.
        et : bool
            Extra-trees.
        xgb : bool
            Gradient tree boosting.

        """
        self.kind = kind
        for key in self.classification_methods:
            exec('self.' + key + ' = ' + key)

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.

        """
        return eval('self.' + self.kind + '.fit(X, y)')

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        """
        return eval('self.' + self.kind + '.predict_proba(X)')


class LeavePairOut(BaseCrossValidator):
    """Leave-pair-out cross-validator.

    Airola, A. (2011), 'An experimental comparison of cross-validation
    techniques for estimating the area under the ROC curve', Computational
    Statistics and Data Analysis 55, 1828-44.
    http://dx.doi.org/10.1016/j.csda.2010.11.018.
    """

    def __init__(self, n_splits=-1):
        """Initialize the object.

        Parameters
        ----------
        n_splits : int, default=-1.
            Number of splitting iterations.

        """
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        indices = np.arange(X.shape[0])
        for test_index in self._iter_test_masks(X, y, groups):
            if groups is not None:
                group = [i in groups[test_index] for i in groups]
                train_index = indices[np.logical_not(group)]
            else:
                train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generate boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups).

        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(X.shape[0], dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X, y=None, groups=None):
        neg_ind = [i for i, v in enumerate(y) if v == 0]
        pos_ind = [i for i, v in enumerate(y) if v == 1]
        combinations = list(it.product(neg_ind, pos_ind))
        random.shuffle(combinations)
        if self.n_splits == -1:
            for combination in combinations:
                yield np.array(combination)
        else:
            for combination in combinations[:self.n_splits]:
                yield np.array(combination)

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        """
        if X is None:
            raise ValueError("The X parameter should not be None")
        neg_ind = [i for i, v in enumerate(y) if v == 0]
        pos_ind = [i for i, v in enumerate(y) if v == 1]
        if self.n_splits == -1:
            return len(list(it.product(neg_ind, pos_ind)))
        else:
            return self.n_splits

def get_pipeline():
    """Generate the pipeline.

    The function generates and returns the following 5-step pipeline:
    1. Unsupervised feature-group selection.
    2. Feature scaling.
    3. Sampling.
    4. Supervised feature selection.
    5. Classification.
    """
    pipe = Pipeline([
        ('inlier_detection', InlierDetection()),
        ('mww_fs', MWWFeatureSelection()), #  remove correlated features
        # ('variance_th', VarianceThreshold()), # remove features with zero-variance
        ('scaler', StandardScaler()), # scale the features
        ('sampling', Sampling(
            kind='none',
            cc=ClusterCentroids(),
            cnn=CondensedNearestNeighbour(),
            enn=EditedNearestNeighbours(),
            renn=RepeatedEditedNearestNeighbours(),
            aknn=AllKNN(),
            iht=InstanceHardnessThreshold(),
            nm=NearMiss(),
            ncr=NeighbourhoodCleaningRule(),
            oss=OneSidedSelection(),
            rus=RandomUnderSampler(),
            tl=TomekLinks(),
            adasyn=ADASYN(),
            ros=RandomOverSampler(),
            smote=SMOTE(),
            smoteenn=SMOTEENN(),
            smotetomek=SMOTETomek(),
        )),
        ('fs', FeatureSelection(
            kind='none',
            k=2,
            ufs_anova=SelectKBest(score_func=f_classif),
            ufs_mi=SelectKBest(score_func=mutual_info_classif),
            rfe_lr=RFE(LogisticRegression(penalty='l2', solver='liblinear')),
            rfe_et=RFE(ExtraTreesClassifier(), step=0.5),
            mb_lr=SelectKBestFromModel(LogisticRegression(penalty='l1', solver='liblinear')),
            mb_et=SelectKBestFromModel(ExtraTreesClassifier()),
            mb_xgb=SelectKBestFromModel(XGBClassifier(silent=True, n_jobs=1)),
            relieff=ReliefF(),
            surf=SURF(),
            surfs=SURFstar(),
            msurf=MultiSURF(),
            msurfs=MultiSURFstar(),
        )),
        ('clf', Classification(
            kind='lr2',
            lr1=LogisticRegression(penalty='l1', n_jobs=1),
            lr2=LogisticRegression(penalty='l2', solver='liblinear', n_jobs=1),
            sgd=SGDClassifier(penalty='elasticnet', max_iter=2000, n_jobs=1),
            knn=KNeighborsClassifier(n_jobs=1),
            svm=SVC(kernel='rbf', probability=True),
            et=ExtraTreesClassifier(n_jobs=1),
            xgb=XGBClassifier(silent=True, n_jobs=1),
        )),
    ])
    return pipe


def get_param_dist():
    """Define hyperparameter distribution.

    This function defines and returns hyperparameter distributions which are
    sampled in model tuning.
    """
    param_dist = {
        # Inlier detection
        'inlier_detection__method': ['none', 'isolation_forest'],
        # Correlation
        # 'correlation_th__threshold': np.arange(0.5, 1.01, 0.05),
        # MWW feature selection
        'mww_fs__corr_method': ['pearson', 'spearman'],
        'mww_fs__corr_threshold': sp_uniform(),
        'mww_fs__n_feats': sp_uniform(),
        # Sampling
        'sampling__kind': ['none', 'cc', 'cnn', 'enn', 'renn', 'aknn',
                           'iht', 'nm', 'ncr', 'oss', 'rus', 'tl', 'adasyn',
                           'ros', 'smote', 'smoteenn', 'smotetomek'],
        'sampling__enn__n_neighbors': sp_randint(2, 3, 5),
        'sampling__enn__kind_sel': ('all', 'mode'),
        'sampling__renn__n_neighbors': [2, 3, 5],
        'sampling__aknn__n_neighbors': [2, 3, 5],
        'sampling__aknn__kind_sel': ('all', 'mode'),
        'sampling__iht__cv': [5, 10],
        'sampling__nm__n_neighbors': [2, 3, 5],
        'sampling__nm__n_neighbors_ver3': [2, 3, 5],
        'sampling__ncr__n_neighbors': [2, 3, 5],
        'sampling__adasyn__n_neighbors': [3, 5, 8],
        'sampling__smote__k_neighbors': [3, 4, 5],
        'sampling__smote__m_neighbors': [7, 8, 9],
        'sampling__smote__kind': ('regular', 'borderline1', 'borderline2'),
        # Scaler
        'scaler': (StandardScaler(), RobustScaler()),
        # Feature Selection
        'fs__kind': ('none', 'ufs_anova', 'ufs_mi', 'rfe_lr', 'rfe_et',
                     'mb_lr', 'mb_et', 'mb_xgb', 'relieff', 'surf', 'surfs', 'msurf',
                     'msurfs'),
        'fs__k': sp_randint(2, 6),
        'fs__rfe_lr__estimator__C': np.logspace(-5, 10, 1000, base=2),
        'fs__rfe_lr__estimator__class_weight': [None, 'balanced'],
        'fs__rfe_et__estimator__n_estimators': range(90, 140),
        'fs__rfe_et__estimator__class_weight': [None, 'balanced',
                                                'balanced_subsample'],
        'fs__mb_lr__estimator__penalty': ('l1', 'l2'),
        'fs__mb_lr__estimator__C': np.logspace(-5, 10, 1000, base=2),
        'fs__mb_lr__estimator__class_weight': [None, 'balanced'],
        'fs__mb_et__estimator__n_estimators': range(90, 140),
        'fs__mb_et__estimator__class_weight': [None, 'balanced',
                                               'balanced_subsample'],
        'fs__mb_xgb__estimator__learning_rate': np.logspace(-7, -1, 1000, base=2),
        'fs__mb_xgb__estimator__gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        'fs__mb_xgb__estimator__max_depth': sp_randint(1, 6),
        'fs__mb_xgb__estimator__min_child_weight': [1, 3, 5, 7],
        'fs__mb_xgb__estimator__subsample': [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                                1],
        'fs__mb_xgb__estimator__reg_lambda': sp_uniform(),
        'fs__mb_xgb__estimator__reg_alpha': sp_uniform(),
        'fs__mb_xgb__estimator__n_estimators': range(200, 2001),
        # Classification
        'clf__kind': ['lr1', 'lr2', 'sgd', 'knn', 'svm', 'et', 'xgb'],
        'clf__lr1__class_weight': (None, 'balanced'),
        'clf__lr1__C': np.logspace(-5, 10, 1000, base=2),
        'clf__lr2__class_weight': (None, 'balanced'),
        'clf__lr2__C': np.logspace(-5, 10, 1000, base=2),
        'clf__sgd__class_weight': (None, 'balanced'),
        'clf__sgd__l1_ratio': sp_uniform(),
        'clf__sgd__alpha': np.logspace(-5, 10, 1000, base=0.5),
        'clf__sgd__loss': ['log', 'log'],
        'clf__knn__p': (1, 2, float('inf')),
        'clf__knn__n_neighbors': sp_randint(1, 9),
        'clf__svm__class_weight': (None, 'balanced'),
        'clf__svm__C': np.logspace(-5, 10, 1000, base=2),
        'clf__svm__gamma': np.logspace(-15, 3, 1000, base=2),
        'clf__et__max_features': np.arange(0.05, 1.01, 0.05),
        'clf__et__criterion': ['gini', 'entropy'],
        'clf__et__min_samples_split': range(2, 21),
        'clf__et__min_samples_leaf': range(1, 21),
        'clf__et__n_estimators': range(100, 300),
        'clf__et__class_weight': [None, 'balanced', 'balanced_subsample'],
        'clf__xgb__learning_rate': np.logspace(-7, -1, 1000, base=2),
        'clf__xgb__gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
        'clf__xgb__max_depth': sp_randint(1, 6),
        'clf__xgb__min_child_weight': [1, 3, 5, 7],
        'clf__xgb__subsample': [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                                1],
        'clf__xgb__reg_lambda': sp_uniform(),
        'clf__xgb__reg_alpha': sp_uniform(),
        'clf__xgb__n_estimators': range(200, 2001),
    }
    return param_dist


def get_performance(rand_search):
    """Calculate performance metrics based on model tuning scores.

    The function calculates standard deviation and confidence intervals based
    on cross-validation performance scores from model tuning.

    """
    auc = rand_search.best_score_
    test_scores = [value[rand_search.best_index_] for key, value in
                   rand_search.cv_results_.items() if
                   (key.startswith("split") and key.endswith("test_score"))]
    std = np.std(test_scores)
    if len(test_scores) >= 30:
        try:
            ci = bootstrap.ci(test_scores)
        except IndexError:
            ci = [np.nan, np.nan]
    else:
        ci = [np.nan, np.nan]
    return auc, std, ci, test_scores


def tune_model(x_this, y_this, params):
    """Tune models.

    Models are tuned according to parameters specified in `params` dictionary.

    Parameters
    ----------
    params['scalers'] : list, {'standard', 'robust'}.
        Feature scaling algorithm.

    params['sampling'] : list, {'none', 'cc', 'cnn', 'enn', 'renn',
                                      'aknn', 'iht', 'nm', 'ncr', 'oss', 'rus',
                                      'tl', 'adasyn', 'ros', 'smote',
                                      'smoteenn', 'smotetomek'}.
        Sampling method(s) to tune.

    params['fs'] : list, {'none', 'ufs_anova', 'ufs_mi', 'rfe_lr',
                                       'rfe_et', 'mb_lr', 'mb_et', 'relieff',
                                       'surf', 'surfs', 'msurf', 'msurfs'}.
        Feature selection method(s) to tune.

    params['clf'] : list, {'lr1', 'lr2', 'sgd', 'knn', 'svm',
                                        'et', 'xgb'}.
        Classifier(s) to tune.

    params['cv_kinds'] : str
        Type of cross-validation to use.
        'sss' - Stratified shuffle split
        'lpo' - leave-pair-out

    params['n_iters'] : list
        Number of iterations, i.e. hyperparameter samples, in hyperparameter
        tuning.

    params['cv_reps'] : list
        Number of iterations in cross-validation.

    params['n_jobs'] : int
        Number of threads in multiprocessing.

    params['random_state'] : int
        Seed for reproducibility of results.

    params['df_pickle_name'] : str
        Path of the binary pickle with the results.

    """

    def evaluate_model_candidate(clf, fs, sampling):
        """Evaluate specific model candidate.

        Upon finished evaluation it saves a pickle with a result.

        Parameters
        ----------
        clf : string
            Classifier.
        fs : string
            Feature selection algorithm.
        sampling : string
            Sampling algorithm.

        Returns
        -------
        RandomizedSearchCV object.

        """
        # Get the default pipeline
        pipe = get_pipeline()

        # Get the default hyperparameter distributions
        param_dist = get_param_dist()

        # Inlier selection
        param_dist['inlier_detection__method'] = params['inlier_detection']

        # Scaling
        if params['scaler'] == 'standard':
            param_dist['scaler'] = [StandardScaler(), StandardScaler()]
        elif params['scaler'] == 'robust':
            param_dist['scaler'] = [RobustScaler(), RobustScaler()]
        # Sampling
        param_dist['sampling__kind'] = [sampling]
        # Feature selection
        param_dist['fs__kind'] = [fs]
        # Classification
        param_dist['clf__kind'] = [clf]
        # Cross-validation
        if params['cv_kind'] == 'sss':
            cv_kind = StratifiedShuffleSplit(
                n_splits=params['cv_splits'], test_size=0.1,
                random_state=params['random_state'])
        elif params['cv_kind'] == 'rskf':
            cv_kind = RepeatedStratifiedKFold(
                n_splits=params['cv_splits'],
                n_repeats=params['cv_repeats'],
                random_state=params['random_state'])
        elif params['cv_kind'] == 'lpo':
            cv_kind = LeavePairOut(n_splits=params['cv_splits'])
        else:
            cv_kind = 10

        # Initialize a RandomizedSearchCV instance
        rand_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            scoring='roc_auc',
            cv=cv_kind,
            n_iter=params['rs_iters'],
            verbose=0,
            random_state=params['random_state'],
            n_jobs=params['n_jobs'],
        )

        start_time = time.time()
        rand_search.fit(x_this, y_this)
        comp_time = time.time() - start_time

        y_pred = rand_search.predict_proba(x_this)
        auc_train = roc_auc_score(y_true=y_this, y_score=y_pred[:, 1])
        auc_valid, std_valid, ci_valid, _ = get_performance(rand_search)
        df.loc[i, 'clf'] = rand_search.best_params_['clf__kind']
        df.loc[i, 'fs'] = rand_search.best_params_['fs__kind']
        df.loc[i, 'sampling'] = rand_search.best_params_['sampling__kind']
        df.loc[i, 'scaler'] = params['scaler']
        df.loc[i, 'cv_kind'] = params['cv_kind']
        df.loc[i, 'cv_splits'] = params['cv_splits']
        df.loc[i, 'cv_repeats'] = params['cv_repeats']
        df.loc[i, 'rs_iters'] = params['rs_iters']
        df.loc[i, 'n_jobs'] = params['n_jobs']
        df.loc[i, 'train_auc'] = auc_train
        df.loc[i, 'valid_auc'] = auc_valid
        df.loc[i, 'valid_std'] = std_valid
        df.loc[i, 'valid_auc-std'] = auc_valid - std_valid
        df.loc[i, 'valid_ci'] = str(ci_valid)
        df.loc[i, 'valid_ci_l'] = ci_valid[0]
        df.loc[i, 'valid_ci_h'] = ci_valid[1]
        df.loc[i, 'valid_ci_width'] = ci_valid[1]-ci_valid[0]
        df.loc[i, 'valid_time_[s]'] = np.round(comp_time)
        df.loc[i, 'model'] = rand_search
        df.to_pickle(params['df_pickle_name'])
        return rand_search

    params = copy.deepcopy(params)
    try:
        df = pd.read_pickle(params['df_pickle_name'])
    except FileNotFoundError:
        df = pd.DataFrame()
    i = 0
    purge_no = 1
    last_purge = 0
    print('Date and time: classifier - feature selection - sampling')
    while (len(params['clf_kinds'])
           * len(params['fs_kinds'])
           * len(params['sampling_kinds']) > 1):
        print('Random search interations: {}'.format(params['rs_iters']))
        for clf in params['clf_kinds']:
            for fs in params['fs_kinds']:
                for sampling in params['sampling_kinds']:
                    print('{}: {} - {} - {}'.format(time.asctime(), clf, fs,
                                                    sampling))
                    if i not in df.index:
                        evaluate_model_candidate(clf, fs, sampling)
                    i += 1
        print(30*'#')
        print('Purge {}'.format(purge_no))
        for algorithm in ['clf', 'fs', 'sampling']:
            if len(params[algorithm + '_kinds']) > 1:
                df_perf = (df.loc[last_purge:i-1, :].groupby(algorithm)
                           .median()[['valid_auc', 'valid_std',
                                      'valid_auc-std']]
                           .sort_values('valid_auc'))
                print(df_perf)
                to_remove = df_perf.index[0]
                params[algorithm + '_kinds'].remove(to_remove)
                print('Removed: {}'.format(to_remove))
                print('Left: {}'.format(params[algorithm + '_kinds']))
                params['rs_iters'] *= ((len(params[algorithm + '_kinds']) + 1)
                                       / len(params[algorithm + '_kinds']))
                params['rs_iters'] = int(params['rs_iters'])
        purge_no += 1
        last_purge = i
    print('Random search interations: {}'.format(params['rs_iters']))
    print('{}: {} - {} - {}'.format(time.asctime(), params['clf_kinds'],
                                    params['fs_kinds'],
                                    params['sampling_kinds']))
    if i not in df.index:
        model = evaluate_model_candidate(clf, fs, sampling)
    else:
        model = df.loc[i, 'model']
    return model


def test_model(X, y, params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    auc_scores = list()
    fprs = list()
    tprs = list()
    i = 1
    params = copy.deepcopy(params)
    params['df_pickle_name'] = params['df_pickle_name'][:-2] + '_split_0.p'
    for train_index, test_index in skf.split(X, y):
        # split the data to train and test splits
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # tune the model on the training split
        params['df_pickle_name'] = (params['df_pickle_name'][:-3]
                                    + str(i) + '.p')
        model = tune_model(X_train, y_train, params)
        # predict on the test split
        y_pred = model.predict_proba(X_test)
        # evalute the performance
        auc = roc_auc_score(y_test, y_pred[:, 1])
        print('Test AUC: {}'.format(auc))
        auc_scores.append(auc)
        #  calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        fprs.append(fpr)
        tprs.append(tpr)

        i += 1
    return auc_scores, fprs, tprs
