import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters_perm, clusters_letter, get_cluster_data
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, decomposition
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt
import sys
import os

if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('plots'):
    os.mkdir('plots')

out = './results'

def run_km_em(perm_x, perm_y, dname, clstr):
    SSE_km_perm = []
    ll_em_perm = []  
    acc_km_perm = []
    acc_em_perm = []
    adjMI_km_perm = []
    adjMI_em_perm = []
    homo_km_perm = []
    homo_em_perm = []
    comp_km_perm = []
    comp_em_perm = []
    silhou_km_perm = []
    bic_em_perm = []
    clk_time = []

    for k in clstr:
        st = clock()
        km = KMeans(n_clusters=k, random_state=10)
        gmm = GMM(n_components=k, random_state=10)

        SSE_km_perm.append(-km.score(perm_x, km.fit_predict(perm_x)))
        ll_em_perm.append(gmm.score(perm_x, gmm.fit_predict(perm_x)))
        acc_km_perm.append(cluster_acc(perm_y,km.fit_predict(perm_x)))
        acc_em_perm.append(cluster_acc(perm_y,gmm.fit_predict(perm_x)))
        adjMI_km_perm.append(ami(perm_y,km.fit_predict(perm_x)))
        adjMI_em_perm.append(ami(perm_y,gmm.fit_predict(perm_x)))
        homo_km_perm.append(metrics.homogeneity_score(perm_y,km.fit_predict(perm_x)))
        homo_em_perm.append(metrics.homogeneity_score(perm_y,gmm.fit_predict(perm_x)))
        comp_km_perm.append(metrics.completeness_score(perm_y,km.fit_predict(perm_x)))
        comp_em_perm.append(metrics.completeness_score(perm_y,gmm.fit_predict(perm_x)))
        silhou_km_perm.append(metrics.silhouette_score(perm_x,km.fit_predict(perm_x)))
        bic_em_perm.append(gmm.bic(perm_x))
        clk_time.append(clock()-st)
        print(k, clock()-st)
        
    dbcluster = pd.DataFrame({
                'k' : clstr,
                'SSE_km' : SSE_km_perm,
                'll_em' : ll_em_perm ,
                'acc_km' : acc_km_perm,
                'acc_em' : acc_em_perm,
                'adjMI_km' : adjMI_km_perm,
                'adjMI_em' : adjMI_em_perm,
                'homo_km' : homo_km_perm,
                'homo_em' : homo_em_perm,
                'comp_km' : comp_km_perm,
                'comp_em' : comp_em_perm,
                'silhou_km' : silhou_km_perm,
                'bic_em' : bic_em_perm,
                'clk_time' : clk_time})

    dbcluster.to_csv('./results/cluster_{}.csv'.format(dname), sep=',')

def plot_km_em(dname):
    db = pd.read_csv('./results/cluster_{}.csv'.format(dname))

    if dname=='Visa':
        title = ['SSE KMeans Visa', 'Log Probability GMM Visa', 'ACC KMeans Visa', 'ACC GMM Visa', 'AMI KMeans Visa',
            'AMI GMM Visa', 'Homogeneity Score KMeans Visa', 'Homogeneity Score GMM Visa', 'Completeness Score KMeans Visa', 
            'Completeness Score GMM Visa', 'Visa Avg KMeans Silhouette Score', 'Visa GMM BIC']

        yval = ['SSE_km', 'll_em', 'acc_km', 'acc_em', 'adjMI_km',
            'adjMI_em', 'homo_km', 'homo_em', 'comp_km', 'comp_em','silhou_km', 'bic_em']

        ylbl = ['SSE', 'Log Probability', 'ACC', 'ACC', 'AMI',
            'AMI', 'Homogeneity Score', 'Homogeneity Score', 'Completeness Score', 
            'Completeness Score', 'Silhouette Score', 'BIC']
    else:
        title = ['SSE KMeans Letter Recognition', 'Log Probability GMM Letter Recognition', 
            'ACC KMeans Letter Recognition', 'ACC GMM Letter Recognition', 'AMI KMeans Letter Recognition', 'AMI GMM Letter Recognition', 
            'Homogeneity Score KMeans Letter Recognition', 'Homogeneity Score GMM Letter Recognition', 
            'Completeness Score KMeans Letter Recognition', 'Completeness Score GMM Letter Recognition',
            'Letter Recognition Avg KMeans Silhouette Score', 'Letter Recognition GMM BIC']

        yval = ['SSE_km', 'll_em', 'acc_km', 'acc_em', 'adjMI_km',
            'adjMI_em', 'homo_km', 'homo_em', 'comp_km', 'comp_em','silhou_km', 'bic_em']

        ylbl = ['SSE', 'Log Probability', 
            'ACC', 'ACC', 'AMI', 'AMI', 'Homogeneity Score', 'Homogeneity Score', 
            'Completeness Score', 'Completeness Score', 'Silhouette Score', 'BIC']

    for t, yv, yl in zip(title, yval, ylbl):
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(db['k'], db[yv], '*-', color='b')
        plt.xlabel('Number of Clusters')
        plt.ylabel(yl)
        plt.title(t)
        plt.grid()
        plt.savefig('./plots/{}_{}.png'.format(yv, dname))
        plt.close()          

def plot_combined(dname, dr_name):
    db = pd.read_csv('./results/cluster_{}.csv'.format(dname))
    db2d = pd.read_csv('./results/{}_{}_2D.csv'.format(dname, dr_name))
    x1 = db2d['x1']
    x2 = db2d['x2']
    km = db2d['km']
    gmm = db2d['gmm']

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))

    # plot SSE for K-Means
    k = db['k']
    metric = db['SSE_km']
    ax1.plot(k, metric, marker='o', markersize=5, color='g')
    ax1.set_title('K-Means SSE ({})'.format(dname))
    ax1.set_ylabel('Sum of squared error')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')

    # plot Silhoutte Score for K-Means
    metric = db['silhou_km']
    ax2.plot(k, metric, marker='o', markersize=5, color='b')
    ax2.set_title('K-Means Avg Silhouette Score ({})'.format(dname))
    ax2.set_ylabel('Mean silhouette score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')

    # plot log-likelihood for EM
    metric = db['ll_em']
    ax3.plot(k, metric, marker='o', markersize=5, color='r')
    ax3.set_title('GMM Log-likelihood ({})'.format(dname))
    ax3.set_ylabel('Log-likelihood')
    ax3.set_xlabel('Number of clusters (k)')
    ax3.grid(color='grey', linestyle='dotted')

    # plot BIC for EM
    metric = db['bic_em']
    ax4.plot(k, metric, marker='o', markersize=5, color='b')
    ax4.set_title('GMM BIC ({})'.format(dname))
    ax4.set_ylabel('BIC')
    ax4.set_xlabel('Number of clusters (k)')
    ax4.grid(color='grey', linestyle='dotted')

    ax5.scatter(x1, x2, marker='x', s=20, c=km, cmap='gist_rainbow')
    ax5.set_title('K-Means Clusters ({})'.format(dname))
    ax5.set_ylabel('x1')
    ax5.set_xlabel('x2')
    ax5.grid(color='grey', linestyle='dotted')

    ax6.scatter(x1, x2, marker='x', s=20, c=gmm, cmap='gist_rainbow')
    ax6.set_title('GMM Clusters ({})'.format(dname))
    ax6.set_ylabel('x1')
    ax6.set_xlabel('x2')
    ax6.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('./plots/ClusteringCombined_{}.png'.format(dname)) 


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances

def nn_cluster_datasets(X, dname, km_k, gmm_k):
    """Generates datasets for ANN classification by appending cluster label to
    original dataset.

    Args:
        X (Numpy.Array): Original attributes.
        name (str): Dataset name.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.

    """
    km = KMeans(random_state=10).set_params(n_clusters=km_k)
    gmm = GMM(random_state=10).set_params(n_components=gmm_k)
    km.fit(X)
    gmm.fit(X)

    # add cluster labels to original attributes
    km_x = np.concatenate((X, km.labels_[:, None]), axis=1)
    gmm_x = np.concatenate((X, gmm.predict(X)[:, None]), axis=1)

    # save results
    np.savetxt('./results/{}_km_labels.csv'.format(dname), km_x,  delimiter=',', fmt='%.20f')
    np.savetxt('./results/{}_gmm_labels.csv'.format(dname), gmm_x, delimiter=',', fmt='%.20f')


if __name__ == "__main__":
    perm_x, perm_y, letter_x, letter_y = load_data()

    run_km_em(perm_x, perm_y, 'Visa', clusters_perm)
    run_km_em(letter_x, letter_y, 'Letter', clusters_letter)

    plot_km_em('Visa')
    plot_km_em('Letter')    

    get_cluster_data(perm_x, perm_y, 'Visa', 'Cluster', 6, 6)
    get_cluster_data(letter_x, letter_y, 'Letter', 'Cluster', 30, 30)

    plot_combined('Visa', 'Cluster')
    plot_combined('Letter', 'Cluster')

    nn_cluster_datasets(perm_x, 'Visa', 4, 4)
    nn_cluster_datasets(letter_x, 'Letter', 30, 30)