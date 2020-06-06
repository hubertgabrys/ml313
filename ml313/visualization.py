import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def plot_roc_curve(df, column, y):
    pipe = Pipeline(steps=[('scaler', RobustScaler()), ('clf', LogisticRegression())])
    y_score = pipe.fit(df.loc[:, column].values.reshape(-1, 1), y).decision_function(
        df.loc[:, column].values.reshape(-1, 1))
    fpr, tpr, _ = metrics.roc_curve(y, y_score)
    roc_auc = metrics.auc(fpr, tpr)
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
    return fpr, tpr, roc_auc


def plot_auc_vs_wavelet(df_auc, feat2flip, rownames, colnames):
    arr_foo = df_auc.loc[feat2flip, 'AUC'].values.reshape(9, 154).T
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.imshow(arr_foo)
    plt.yticks(range(154), rownames)
    plt.xticks(range(9), colnames, rotation='vertical')
    ax.xaxis.tick_top()
    ax.set_xlabel('Wavelet transformation')
    ax.set_ylabel('Feature')
    plt.colorbar()
    plt.show()
