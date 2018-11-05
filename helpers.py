import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv
from collections import defaultdict
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from time import clock
from sklearn import preprocessing
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt

nn_layers = [(100,), (50,), (50, 50)]
nn_reg = [10**-x for x in range(1,5)]
nn_iter = 1500

clusters_perm =  [2,3,4,5,7,10,15,20,25,30]
clusters_letter =  [2,3,4,5,7,10,15,20,23,25,27,30,35,40,45,50,55,60,65,70]

dims_perm = [2,3,4,5,6,7]
dims_letter = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def load_data():
	np.random.seed(0)


	perm = pd.read_csv("./data/visa.csv")
	perm_Y_VAL = 'case_status'

	letter = pd.read_csv('./data/letter-recognition.data')
	features_to_encode = ['lettr']
	le = preprocessing.LabelEncoder
	encoderDict = defaultdict(le)

	for column in features_to_encode:
		letter[column] = letter[column].dropna()
		letter = letter[letter[column].notnull()]
		letter[column] = encoderDict[column].fit_transform(letter[column])

	letter_Y_VAL = 'lettr'


	perm_y = perm[perm_Y_VAL].copy().values
	perm_x = perm.drop(perm_Y_VAL, 1).copy().values

	letter_y = letter[letter_Y_VAL].copy().values
	letter_x = letter.drop(letter_Y_VAL, 1).copy().values


	perm_x = StandardScaler().fit_transform(perm_x)
	letter_x = StandardScaler().fit_transform(letter_x)

	return (perm_x, perm_y, letter_x, letter_y)


def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)

def balanced_accuracy(labels, predictions):
	weights = compute_sample_weight('balanced', labels)
	return accuracy_score(labels, predictions, sample_weight=weights)

def run_clustering(dr_name, dname, perm_x, perm_y, clstrs):
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

	for k in clstrs:
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
                'k' : clstrs,
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

	dbcluster.to_csv('./results/cluster_{}_{}.csv'.format(dr_name, dname), sep=',')

	db = pd.read_csv('./results/cluster_{}_{}.csv'.format(dr_name, dname))
	
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))

    # plot SSE for K-Means
	k = db['k']
	metric = db['SSE_km']
	ax1.plot(k, metric, marker='o', markersize=5, color='g', label='After {}'.format(dr_name))
	ax1.set_title('K-Means SSE ({})'.format(dname))
	ax1.set_ylabel('Sum of squared error')
	ax1.set_xlabel('Number of clusters (k)')
	ax1.grid(color='grey', linestyle='dotted')

	# plot Silhoutte Score for K-Means
	metric = db['silhou_km']
	ax2.plot(k, metric, marker='o', markersize=5, color='b', label='After {}'.format(dr_name))
	ax2.set_title('K-Means Avg Silhouette Score ({})'.format(dname))
	ax2.set_ylabel('Mean silhouette score')
	ax2.set_xlabel('Number of clusters (k)')
	ax2.grid(color='grey', linestyle='dotted')

	# plot log-likelihood for EM
	metric = db['ll_em']
	ax3.plot(k, metric, marker='o', markersize=5, color='r', label='After {}'.format(dr_name))
	ax3.set_title('GMM Log-likelihood ({})'.format(dname))
	ax3.set_ylabel('Log-likelihood')
	ax3.set_xlabel('Number of clusters (k)')
	ax3.grid(color='grey', linestyle='dotted')

	# plot BIC for EM
	metric = db['bic_em']
	ax4.plot(k, metric, marker='o', markersize=5, color='b', label='After {}'.format(dr_name))
	ax4.set_title('GMM BIC ({})'.format(dname))
	ax4.set_ylabel('BIC')
	ax4.set_xlabel('Number of clusters (k)')
	ax4.grid(color='grey', linestyle='dotted')

	# change layout size, font size and width between subplots
	fig.tight_layout()
	for ax in fig.axes:
		ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
		for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
			item.set_fontsize(8)
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('./plots/ClusteringCombined_{}_{}.png'.format(dr_name, dname)) 

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

    
def aveMI(X,Y):    
    MI = MIC(X,Y) 
    return np.nanmean(MI)
    
  
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def get_cluster_data(X, y, name, dr_name, km_k, gmm_k, perplexity=30):
    """Generates 2D dataset that contains cluster labels for K-Means and GMM,
    as well as the class labels for the given dataset.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        perplexity (int): Perplexity parameter for t-SNE.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.
        rdir (str): Folder to save results CSV.

    """
    # generate 2D X dataset
    X2D = TSNE(n_iter=1000, perplexity=perplexity).fit_transform(X)

    # get cluster labels using best k
    km = KMeans(random_state=5).set_params(n_clusters=km_k)
    gmm = GMM(random_state=5).set_params(n_components=gmm_k)
    km_cl = np.atleast_2d(km.fit(X2D).labels_).T
    gmm_cl = np.atleast_2d(gmm.fit(X2D).predict(X2D)).T
    y = np.atleast_2d(y).T

    # create concatenated dataset
    cols = ['x1', 'x2', 'km', 'gmm', 'class']
    df = pd.DataFrame(np.hstack((X2D, km_cl, gmm_cl, y)), columns=cols)

    # save as CSV
    df.to_csv('./results/{}_{}_2D.csv'.format(name, dr_name), sep=',')
        
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
