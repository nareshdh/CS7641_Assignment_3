import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './results/'

def run_rf(X, y, dname, theta):
    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=4)
    fi = rfc.fit(X, y).feature_importances_

    # get feature importance and sort by value in descending order
    i = [i + 1 for i in range(len(fi))]
    fi = pd.DataFrame({'importance': fi, 'feature': i})
    fi.sort_values('importance', ascending=False, inplace=True)
    fi['i'] = i
    cumfi = fi['importance'].cumsum()
    fi['cumulative'] = cumfi

    # generate dataset that meets cumulative feature importance threshold
    idxs = fi.loc[:cumfi.where(cumfi > theta).idxmin(), :]
    idxs = list(idxs.index)
    reduced = X[:, idxs]

    # save results as CSV
    np.savetxt('./results/{}_RF_projected.csv'.format(dname), reduced,  delimiter=',', fmt='%.20f')
    fi.to_csv(out+'rf_{}.csv'.format(dname), index_label=None)


def plot_rf(dname, theta):
    df = pd.read_csv('./results/rf_{}.csv'.format(dname))

    # get figure and axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # plot explained variance and cumulative explain variance ratios
    ax2 = ax1.twinx()
    x = df['i']
    fi = df['importance']
    cumfi = df['cumulative']
    ax1.bar(x, height=fi, color='b', tick_label=df['feature'], align='center')
    ax2.plot(x, cumfi, color='r', label='Cumulative Info Gain')
    ax1.set_title('Feature Importance ({})'.format(dname))
    ax1.set_ylabel('Gini Gain')
    ax2.set_ylabel('Cumulative Gini Gain')
    ax1.set_xlabel('Feature Index')
    ax2.axhline(y=theta, linestyle='--', color='r')
   
    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plt.savefig('./plots/{}_rf.png'.format(dname))

def calc_reconstruction_error(X, y, dname, dims):
    error = []
    idx = []
    for i in dims:
        rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=4)
        filtr = ImportanceSelect(rfc,i)
        filtr.fit_transform(X, y)    
        error.append(reconstructionError(filtr, X))
        print(reconstructionError(filtr, X))
        idx.append(i)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(idx, error, '*-', color="b", label='Error')
    plt.xlabel('Independent Components')
    plt.ylabel('Reconstruction Error')
    plt.title('{} ICA Reconstruction Error'.format(dname))
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('./plots/{}_rf_reconstruction.png'.format(dname))
    plt.close()    

def rerun_RFcluster(X, y, n, clstrs, dname, dr_name):
    dim = n
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=4)
    filtr = ImportanceSelect(rfc,dim)
    perm_x2 = filtr.fit_transform(X, y)

    run_clustering(dr_name, dname, perm_x2, y, clstrs)


if __name__ == '__main__':
    perm_x, perm_y, letter_x, letter_y = load_data()
    theta = 0.80
    run_rf(perm_x, perm_y, 'Visa', theta)
    run_rf(letter_x, letter_y, 'Letter', theta)

    plot_rf('Visa', theta)
    plot_rf('Letter', theta)

    ##calc_reconstruction_error(perm_x, perm_y, 'Visa', dims_perm)
    ##calc_reconstruction_error(letter_x, letter_y, 'Letter', dims_letter)

    ##get_cluster_data(perm_x, perm_y, 'Visa', 'RF', 3, 3)
    ##get_cluster_data(letter_x, letter_y, 'Letter', 'RF', 30, 30)

    rerun_RFcluster(perm_x, perm_y, 4, clusters_perm, 'Visa', 'RF')
    rerun_RFcluster(letter_x, letter_y, 12, clusters_letter, 'Letter', 'RF')