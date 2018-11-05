
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import *
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
import matplotlib.pylab as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import euclidean_distances

out = './results/'

def run_rp(X, dname, dims):
    tmp = defaultdict(dict)
    for i,dim in product(range(10),dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp =pd.DataFrame(tmp).T
    tmp.to_csv(out+'rp_{}.csv'.format(dname))

    tmp = defaultdict(dict)
    for i,dim in product(range(10),dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)    
        tmp[dim][i] = reconstructionError(rp, X)
    tmp =pd.DataFrame(tmp).T
    tmp.to_csv(out+'rp_{}_reconstruction.csv'.format(dname))


def plot_rp(dname):	
	db = pd.read_csv('./results/rp_{}.csv'.format(dname))
	db_cor = db.copy()
	db_cor['mean'] = db.iloc[:,[x for x in range(1,db.shape[1])]].mean(axis=1)
	db_cor['std'] = db.iloc[:,[x for x in range(1,db.shape[1])]].std(axis=1)
	db_cor.rename(columns={'Unnamed: 0':'seed'}, inplace=True)

	db = pd.read_csv('./results/rp_{}_reconstruction.csv'.format(dname))
	db_rec = db.copy()
	db_rec['mean'] = db.iloc[:,[x for x in range(1,db.shape[1])]].mean(axis=1)
	db_rec['std'] = db.iloc[:,[x for x in range(1,db.shape[1])]].std(axis=1)
	db_rec.rename(columns={'Unnamed: 0':'seed'}, inplace=True)

	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
	ax1.errorbar(db_cor['seed'], db_cor['mean'], yerr=db_cor['std'], capsize=5, elinewidth=2, markeredgewidth=2)
	ax1.set_title('{} Pairwise Distance Correlation'.format(dname))
	ax1.set_ylabel('PDC Error Bar')
	ax1.set_xlabel('Number of Components')
	ax1.grid()

	ax2.errorbar(db_rec['seed'], db_rec['mean'], yerr=db_rec['std'], capsize=5, elinewidth=2, markeredgewidth=2)
	ax2.set_title('{} Reconstruction Error'.format(dname))
	ax2.set_ylabel('Reconstruction Error Bar')
	ax2.set_xlabel('Number of Components')
	ax2.grid()
	fig.tight_layout()
	for ax in fig.axes:
		ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
		for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
			item.set_fontsize(8)
	plt.savefig('./plots/{}_rp.png'.format(dname))

def plot_similarity(X, n, dname):
    rp =  GaussianRandomProjection(n_components=n)
    rp_t = rp.fit_transform(X)

    orig_dist = euclidean_distances(X)
    red_dist = euclidean_distances(rp_t)
    diff_dist = abs(orig_dist - red_dist)

    plt.figure()
    plt.pcolor(diff_dist[0:100, 0:100])
    plt.colorbar()
    plt.title('{} Difference in Distance of Original and Reduced Space'.format(dname))
    plt.savefig('./plots/{}_rp_similarity.png'.format(dname))

def rerun_RPcluster(X, y, n, clstrs, dname, dr_name):
    dim = n
    rp = SparseRandomProjection(n_components=dim, random_state=5)
    perm_x2 = rp.fit_transform(X)

    run_clustering(dr_name, dname, perm_x2, y, clstrs)

def save_rp_results(X, dname, dims):
	rp = SparseRandomProjection(random_state=0, n_components=dims)
	results = rp.fit_transform(X)
	np.savetxt('./results/{}_RP_projected.csv'.format(dname), results,  delimiter=',', fmt='%.20f')


if __name__ == "__main__":
	perm_x, perm_y, letter_x, letter_y = load_data()

	run_rp(perm_x, 'Visa', dims_perm)
	run_rp(letter_x, 'Letter', dims_letter)
	
	plot_rp('Visa')
	plot_rp('Letter')

	plot_similarity(perm_x, 5, 'Visa')
	plot_similarity(letter_x, 15, 'Letter')

	##get_cluster_data(perm_x, perm_y, 'Visa', 'RP', 3, 3)
	##get_cluster_data(letter_x, letter_y, 'Letter', 'RP', 30, 30)

	rerun_RPcluster(perm_x, perm_y, 6, clusters_perm, 'Visa', 'RP')
	rerun_RPcluster(letter_x, letter_y, 12, clusters_letter, 'Letter', 'RP')

	save_rp_results(perm_x, 'Visa', 6)
	save_rp_results(letter_x, 'Letter', 12)
    