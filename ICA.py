import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as plt

out = './results/'

def run_ica(X, dname, dims):
    ica = FastICA(random_state=5, max_iter=5000)
    kurt = {}
    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt) 
    kurt.to_csv(out+'{}_ica.csv'.format(dname))

def plot_ica(dname):
    db = pd.read_csv('./results/{}_ica.csv'.format(dname), names=['n', 'kurt'])
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(db['n'], db['kurt'], '*-', color="g")
    plt.xlabel('Number of Independent Components')
    plt.ylabel('Kurtosis')
    plt.title('{} ICA'.format(dname))
    plt.grid()
    plt.savefig('./plots/{}_ica.png'.format(dname))

def calc_reconstruction_error(X, dname, dims):
	error = []
	idx = []
	for i in dims:
		ica = FastICA(n_components=i, random_state=5)
		ica.fit_transform(X)    
		error.append(reconstructionError(ica, X))
		print(reconstructionError(ica, X))
		idx.append(i)

	ax = plt.figure().gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.plot(idx, error, '*-', color="b", label='Error')
	plt.xlabel('Independent Components')
	plt.ylabel('Reconstruction Error')
	plt.title('{} ICA Reconstruction Error'.format(dname))
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('./plots/{}_ica_reconstruction.png'.format(dname))
	plt.close()    

def rerun_ICAcluster(X, y, n, clstrs, dname, dr_name):
    dim = n
    ica = FastICA(n_components=dim,random_state=5)
    perm_x2 = ica.fit_transform(X)

    run_clustering(dr_name, dname, perm_x2, y, clstrs)

def save_ica_results(X, dname, dims):
	ica = FastICA(random_state=0, max_iter=5000, n_components=dims)
	results = ica.fit_transform(X)
	np.savetxt('./results/{}_ICA_projected.csv'.format(dname), results,  delimiter=',', fmt='%.20f')


if __name__ == "__main__":
	perm_x, perm_y, letter_x, letter_y = load_data()

	run_ica(perm_x, 'Visa', dims_perm)
	run_ica(letter_x, 'Letter', dims_letter)

	plot_ica('Visa')
	plot_ica('Letter')

	save_ica_results(perm_x, 'Visa', dims=5)
	save_ica_results(letter_x, 'Letter', dims=15)

	calc_reconstruction_error(perm_x, 'Visa', dims_perm)
	calc_reconstruction_error(letter_x, 'Letter', dims_letter)

	##get_cluster_data(perm_x, perm_y, 'Visa', 'ICA', 3, 3)
	##get_cluster_data(letter_x, letter_y, 'Letter', 'ICA', 30, 30)

	rerun_ICAcluster(perm_x, perm_y, 6, clusters_perm, 'Visa', 'ICA')
	rerun_ICAcluster(letter_x, letter_y, 12, clusters_letter, 'Letter', 'ICA')