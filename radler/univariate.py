"""
Univariate analysis
"""


# Author: Hubert Gabrys <hubert.gabrys@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from tqdm import tqdm


def mann_whitney_u(X, y, alpha=0.05, validate=False):
    """Computes the Mann-Whitney U test for all columns in X.

    - value of the *U* statistic
    - number of observations in the negative group *n_neg*
    - number of observations in the positive group *n_pos*
    - *p-value* of the test
    - *AUC* calculated based on the U statistic
    - *p-value corrected* for FWER with Bonferroni-Holm procedure
    - *p-value corrected* for FDR with Benjamini-Hochberg procedure

    Parameters
    ----------
    X : Pandas DataFrame, shape (n_observations, n_features)
        Input data.

    y : Pandas Series, shape (n_observations, )
        Array of labels.

    alpha : float, optional
        Significance level for multiple testing correction.

    validate : bool, optional
        Double check of AUC estimation with logistic regression.

    Returns
    -------
    df : Pandas DataFrame, shape (n_features, 7)
    """

    X = pd.DataFrame(X)
    df = pd.DataFrame()
    X_np = np.asarray(X)
    y_np = np.asarray(y) != 0
    for i in tqdm(range(X.shape[1])):
        pos = X_np[y_np, i]
        neg = X_np[~y_np, i]
        pos = pos[~np.isnan(pos.astype(float))]
        neg = neg[~np.isnan(neg.astype(float))]
        n_pos = len(pos)
        n_neg = len(neg)
        try:
            mw_u2, mw_p = mannwhitneyu(pos, neg, alternative='two-sided')
            mw_ubig = max(mw_u2, n_pos*n_neg-mw_u2)
            auc = mw_u2/(n_pos*n_neg)
            # calculate direction
            if mw_u2 < mw_ubig:
                direction = 'negative'
            else:
                direction = 'positive'
            # calculate confidence intervals
            auc_l, auc_h = bootstrap_bca(pos, neg, alpha=boot_alpha, boot_iters=boot_iters)
            if direction == 'negative':
                auc = 1 - auc
                auc_l_old = auc_l
                auc_l = 1 - auc_h
                auc_h = 1 - auc_l_old
        except ValueError:
            # print('Skipping feature because all values are identical in both classes.')
            mw_ubig = np.nan
            mw_p = np.nan
            auc = np.nan
            auc_l = np.nan
            auc_h = np.nan
            direction = np.nan
        # add results to the data frame
        df.loc[X.columns[i], 'U'] = mw_ubig
        df.loc[X.columns[i], 'n_neg'] = n_neg
        df.loc[X.columns[i], 'n_pos'] = n_pos
        df.loc[X.columns[i], 'direction'] = direction
        df.loc[X.columns[i], 'p-value'] = mw_p
        df.loc[X.columns[i], 'AUC'] = auc
        df.loc[X.columns[i], 'AUC_L'] = auc_l
        df.loc[X.columns[i], 'AUC_H'] = auc_h
        if validate:
            # validate with logistic regression
            # Flips like AUC = 1-AUC_lr are due to outliers
            pipe = Pipeline([('scaler', RobustScaler()), ('clf', LogisticRegression())])
            pipe.fit(X_np[:, i].reshape(-1, 1), y_np)
            y_est = pipe.predict_proba(X_np[:, i].reshape(-1, 1))
            auc_lr = metrics.roc_auc_score(y_np, y_est[:, 1])
            df.loc[X.columns[i], 'AUC_lr'] = auc_lr
    # FWER with Bonferroni-Holm procedure
    df['FWER'] = multipletests(df['p-value'], method='h', alpha=0.05)[0]
    # FDR with Benjamini-Hochberg procedure
    df['FDR'] = multipletests(df['p-value'], method='fdr_bh', alpha=0.05)[0]
    # set correct dtypes
    df['n_neg'] = df['n_neg'].astype(int)
    df['n_pos'] = df['n_pos'].astype(int)

    return df


def bootstrap_bca(pos, neg, alpha=0.05):
    n_pos = len(pos)
    n_neg = len(neg)
    auc_b_list = list()
    boot_iters = 1000
    for _ in range(boot_iters):
        this_pos = np.random.choice(pos, n_pos)
        this_neg = np.random.choice(neg, n_neg)
        mw_u2, mw_p = mannwhitneyu(this_pos, this_neg, alternative='two-sided')
        auc_b = mw_u2/(n_pos*n_neg)
        auc_b_list.append(auc_b)
    auc_bs = np.array(auc_b_list)
    # Initial computations
    auc_bs.sort(axis=0)
    mw_u2, mw_p = mannwhitneyu(pos, neg, alternative='two-sided')
    auc_u = mw_u2/(n_pos*n_neg)
    # The bias correction value.
    z0 = norm.ppf(np.sum(auc_bs < auc_u)/boot_iters)    
    # The acceleration value
    jstat = np.zeros(boot_iters)
    for i in range(boot_iters):
        jstat[i] = np.mean(np.delete(auc_bs, i))
    jmean = np.mean(jstat)
    a = np.sum((jmean - jstat)**3) / (6.0 * np.sum((jmean - jstat)**2)**1.5)
    if np.any(np.isnan(a)):
        return np.nan, np.nan
    else:
        # Interval
        alphas = np.array([alpha/2, 1-alpha/2])
        z1 = norm.ppf(alpha/2)
        z2 = norm.ppf(1-alpha/2)
        alpha1 = norm.pdf(z0 + (z0 + z1)/(1-a*(z0+z1)))
        alpha2 = 1 - norm.pdf(z0 + (z0 + z2)/(1-a*(z0+z2)))
        return np.percentile(auc_bs, alpha1*100), np.percentile(auc_bs, alpha2*100)


def recursive_reduction(df_auc, df_corr, threshold, retain, verbose=False):
    df_auc = df_auc.copy()
    df_auc = df_auc.loc[df_auc['FWER']]
    df_auc = df_auc.sort_values('AUC', ascending=False)
    df_corr = df_corr.abs()
    df_corr = df_corr.loc[df_auc.index, df_auc.index]
    i = 1
    feats = list()
    while len(df_auc) > 0:
        if verbose:
            print('Run {}'.format(i))
        if (i == 1) and (retain is not None):
            feats.append(retain)
        else:
            feats.append(df_auc.index[0])
        if verbose:
            print('Best feature: {}'.format(feats[-1]))
        mask = df_corr[feats[-1]] < threshold
        df_auc = df_auc[mask]
        df_corr = df_corr.loc[df_auc.index, df_auc.index]
        i += 1
    return feats


def plot_roc_curve(df, column, y):
    pipe = Pipeline(steps=[('scaler', RobustScaler()), ('clf', LogisticRegression())])
    y_score = pipe.fit(df.loc[:, column].values.reshape(-1, 1), y).decision_function(df.loc[:, column].values.reshape(-1, 1))
    fpr, tpr, _ = metrics.roc_curve(y, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC(' + column + ') = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return fpr, tpr, roc_auc


def plot_auc_vs_wavelet(df_auc, feat2flip, rownames, colnames):
    arr_foo = df_auc.loc[feat2flip, 'AUC'].values.reshape(9, 154).T
    fig, ax = plt.subplots(figsize=(30,30))
    plt.imshow(arr_foo)
    plt.yticks(range(154), rownames)
    plt.xticks(range(9), colnames, rotation='vertical')
    ax.xaxis.tick_top()
    ax.set_xlabel('Wavelet transformation')
    ax.set_ylabel('Feature')
    plt.colorbar()
    plt.show()
