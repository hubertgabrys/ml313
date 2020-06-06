from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from scipy.stats import kruskal
from scipy.stats import f_oneway


def univariate_analysis(X, y, mtc_alpha=0.05, boot_alpha=0.05, boot_iters=2000, n_jobs=1, verbose=0):
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
    :param y: Pandas Series, shape (n_observations, )
        Array of labels.
    :param X: Pandas DataFrame, shape (n_observations, n_features)
        Input data.
    :param mtc_alpha: float, optional
        Significance level for multiple testing correction.
    :param verbose: todo
    :param n_jobs: todo
    :param boot_iters: todo
    :param boot_alpha: todo

    Returns
    -------
    df : Pandas DataFrame, shape (n_features, 7)
    """

    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    def parfor(X_np, y_np, i):
        df_this = pd.DataFrame()
        pos = X_np[y_np == 1, i]
        neg = X_np[y_np == 0, i]
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        n_pos = len(pos)
        n_neg = len(neg)
        median_pos = np.median(pos)
        median_neg = np.median(neg)
        try:
            if (n_neg < 5) or (n_pos < 5):
                raise ValueError('At least one of the samples is too small.')
            cohen_d_value = cohen_d(pos, neg)
            f_stat, f_pval = f_oneway(pos, neg)
            h_stat, h_pval = kruskal(pos, neg)
            mw_u2, mw_p = mannwhitneyu(pos, neg, alternative='two-sided')
            mw_ubig = max(mw_u2, n_pos*n_neg-mw_u2)
            auc = mw_u2/(n_pos*n_neg)
            # calculate the direction
            if mw_u2 < mw_ubig:
                direction = 'negative'
            else:
                direction = 'positive'
            # calculate confidence intervals
            if boot_iters is not None:
                auc_l, auc_h = bootstrap_bca(pos, neg, alpha=boot_alpha, boot_iters=boot_iters)
            else:
                auc_l = np.nan
                auc_h = np.nan
            if direction == 'negative':
                auc = 1 - auc
                auc_l_old = auc_l
                auc_l = 1 - auc_h
                auc_h = 1 - auc_l_old
        except ValueError:
            # print('Skipping feature because all values are identical in both classes.')
            cohen_d_value = np.nan
            f_stat = np.nan
            f_pval = np.nan
            h_stat = np.nan
            h_pval = np.nan
            mw_ubig = np.nan
            mw_p = np.nan
            auc = np.nan
            auc_l = np.nan
            auc_h = np.nan
            direction = np.nan
        # add results to the data frame
        df_this.loc[X.columns[i], 'n_neg'] = n_neg
        df_this.loc[X.columns[i], 'n_pos'] = n_pos
        df_this.loc[X.columns[i], 'median_neg'] = median_neg
        df_this.loc[X.columns[i], 'median_pos'] = median_pos
        df_this.loc[X.columns[i], 'direction'] = direction
        df_this.loc[X.columns[i], 'cohen_d'] = cohen_d_value
        df_this.loc[X.columns[i], 'auc'] = auc
        df_this.loc[X.columns[i], 'auc_cil'] = auc_l
        df_this.loc[X.columns[i], 'auc_cih'] = auc_h
        df_this.loc[X.columns[i], 'u_stat'] = mw_ubig
        df_this.loc[X.columns[i], 'u_pval'] = mw_p
        df_this.loc[X.columns[i], 'f_stat'] = f_stat
        df_this.loc[X.columns[i], 'f_pval'] = f_pval
        df_this.loc[X.columns[i], 'h_stat'] = h_stat
        df_this.loc[X.columns[i], 'h_pval'] = h_pval
        return df_this

    X = pd.DataFrame(X)
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    y_np = y_np.flatten()
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(parfor)(X_np, y_np, i) for i in range(X.shape[1]))
    df = pd.concat(out)
    # FWER with Bonferroni-Holm procedure
    df['FWER'] = multipletests(df['u_pval'], method='h', alpha=mtc_alpha)[0]
    # FDR with Benjamini-Hochberg procedure
    df['FDR'] = multipletests(df['u_pval'], method='fdr_bh', alpha=mtc_alpha)[0]
    # set correct dtypes
    df['n_neg'] = df['n_neg'].astype(int)
    df['n_pos'] = df['n_pos'].astype(int)
    return df


def bootstrap_bca(pos, neg, alpha=0.05, boot_iters=2000):
    n_pos = len(pos)
    n_neg = len(neg)
    auc_b_list = list()
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
        z1 = norm.ppf(alpha/2)
        z2 = norm.ppf(1-alpha/2)
        alpha1 = norm.pdf(z0 + (z0 + z1)/(1-a*(z0+z1)))
        alpha2 = 1 - norm.pdf(z0 + (z0 + z2)/(1-a*(z0+z2)))
        return np.percentile(auc_bs, alpha1*100), np.percentile(auc_bs, alpha2*100)
