import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import *
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt

import pandas as pd

out = './results/'

def run_pca(X, dname, n):
	pca = PCA(random_state=5)
	pca.fit(X)
	tmp = pd.Series(data = pca.explained_variance_,index = range(1,n))
	tmp1 = pd.Series(data = pca.explained_variance_ratio_,index = range(1,n))
	tmp2 = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,n))
	tmp3 = pd.Series(data = pca.singular_values_,index = range(1,n))
	results = pca.fit_transform(X)[:, :tmp2.where(tmp2 > 0.85).idxmin()]
	pd.concat([tmp, tmp1, tmp2, tmp3], axis=1).to_csv(out+'pca_{}.csv'.format(dname))
	np.savetxt('./results/{}_PCA_projected.csv'.format(dname), results,  delimiter=',', fmt='%.20f')

def plot_pca(dname):
	db = pd.read_csv('./results/pca_{}.csv'.format(dname), names=['n', 'ev', 'va', 'cva', 'sv'])
	ax = plt.figure().gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.plot(db['n'], db['ev'], '*-', color="g", label='Eigenvalue')
	plt.plot(db['n'], db['va'], '*-', color="b", label='Variance')
	plt.plot(db['n'], db['cva'], '*-', color="black", label='Cumulative Variance')
	plt.xlabel('Number of Principle Components')
	plt.ylabel('Eigenvalues/Variances')
	plt.title('{} PCA'.format(dname))
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('./plots/{}_pca.png'.format(dname))
	plt.close()

def plot_pca_var(dname, n):
	db = pd.read_csv('./results/pca_{}.csv'.format(dname), skiprows=[0], names=['n', 'ev', 'va', 'cva', 'sv'])
	tot = sum(db['ev'])
	var_exp = [(i / tot)*100 for i in sorted(db['ev'], reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	plt.bar(['%s' %i for i in range(1,n)], var_exp)
	plt.plot(['%s' %i for i in range(1,n)],cum_var_exp, '*-', color="r", label='Cumulative Explained Variance')
	plt.xlabel('Principal Components')
	plt.ylabel('Explained Variance (%)')
	plt.title('{} Explained Variance'.format(dname))
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('./plots/{}_pca_variance.png'.format(dname))
	plt.close()

	# perm_x, perm_y, letter_x, letter_y = load_data()
	# X_std = StandardScaler().fit_transform(perm_x)
	# mean_vec = np.mean(X_std, axis=0)
	# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
	# cov_mat = np.cov(X_std.T)
	# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
	# eig_vals

def calc_reconstruction_error(X, dname, n):
	error = []
	idx = []
	for i in range(1, n):
		rp = PCA(n_components=i)
		rp.fit(X)    
		error.append(reconstructionError(rp, X))
		print(reconstructionError(rp, X))
		idx.append(i)

	ax = plt.figure().gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.plot(idx, error, '*-', color="b", label='Error')
	plt.xlabel('Principle Components')
	plt.ylabel('Reconstruction Error')
	plt.title('{} PCA Reconstruction Error'.format(dname))
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('./plots/{}_pca_reconstruction.png'.format(dname))
	plt.close()
		
def rerun_PCAcluster(X, y, n, clstrs, dname, dr_name):
	dim = n
	pca = PCA(n_components=dim, random_state=5)
	perm_x2 = pca.fit_transform(X)

	run_clustering(dr_name, dname, perm_x2, y, clstrs)
	


if __name__ == "__main__":
	perm_x, perm_y, letter_x, letter_y = load_data()

	run_pca(perm_x, 'Visa', 8)
	run_pca(letter_x, 'Letter', 17)
	
	plot_pca('Visa')
	plot_pca('Letter')
	
	plot_pca_var('Visa', 8)
	plot_pca_var('Letter', 17)
	
	calc_reconstruction_error(perm_x, 'Visa', 7)
	calc_reconstruction_error(letter_x, 'Letter', 16)
	
	##get_cluster_data(perm_x, perm_y, 'Visa', 'PCA', 3, 3)
	##get_cluster_data(letter_x, letter_y, 'Letter', 'PCA', 30, 30)

	rerun_PCAcluster(perm_x, perm_y, 6, clusters_perm, 'Visa', 'PCA')
	rerun_PCAcluster(letter_x, letter_y, 10, clusters_letter, 'Letter', 'PCA')

