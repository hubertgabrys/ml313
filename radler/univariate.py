import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from statsmodels.stats.multitest import multipletests


def mann_whitney_u(X, y, alpha=0.05, validate=False):
    '''The function performs the test and returns:
    - value of the *U* statistic
    - *p-value* of the test
    - *AUC* calculated based on the U statistic
    - *p-value corrected* for FWER with Bonferroni-Holm procedure
    - *p-value corrected* for FDR with Benjamini-Hochberg procedure'''

    X = pd.DataFrame(X)
    df = pd.DataFrame()
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    for i in range(X.shape[1]):
        pos = X_np[y_np, i]
        neg = X_np[~y_np, i]
        pos = pos[~np.isnan(pos)]
        neg = neg[~np.isnan(neg)]
        n_pos = len(pos)
        n_neg = len(neg)
        try:
            mw_u2, mw_p = mannwhitneyu(pos, neg, alternative='two-sided')
            mw_ubig = max(mw_u2, n_pos*n_neg-mw_u2)
            auc = mw_ubig/(n_pos*n_neg)
        except ValueError:
            print('Skipping feature because all values are identical in both classes.')
            mw_ubig = np.nan
            mw_p = np.nan
            auc = np.nan
        # add results to the data frame
        df.loc[X.columns[i], 'U'] = mw_ubig
        df.loc[X.columns[i], 'p-value'] = mw_p
        df.loc[X.columns[i], 'AUC'] = auc
        if validate:
            # validate with logistic regression
            # Flips like AUC = 1-AUC_lr are due to outliers
            pipe = Pipeline([('scaler', RobustScaler()), ('clf', LogisticRegression())])
            pipe.fit(X_np[:, i].reshape(-1, 1), y_np)
            y_est = pipe.predict_proba(X_np[:, i].reshape(-1, 1))
            auc_lr = roc_auc_score(y_np, y_est[:, 1])
            df.loc[X.columns[i], 'AUC_lr'] = auc_lr
    # FWER with Bonferroni-Holm procedure
    df['FWER'] = multipletests(df['p-value'], method='h', alpha=0.05)[0]
    # FDR with Benjamini-Hochberg procedure
    df['FDR'] = multipletests(df['p-value'], method='fdr_bh', alpha=0.05)[0]

    return df


def plot_roc_curve(df, column, y):
    pipe = Pipeline(steps=[('scaler', RobustScaler()), ('clf', LogisticRegression())])
    y_score = pipe.fit(df.loc[:, column].values.reshape(-1, 1), y).decision_function(df.loc[:, column].values.reshape(-1, 1))
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC({}) = {:.2f}'.format(column, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


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
