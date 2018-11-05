import pandas as pd
import matplotlib.pylab as plt

def plot_recluster(dname):
	dbpca = pd.read_csv('./results/cluster_PCA_{}.csv'.format(dname))
	dbica = pd.read_csv('./results/cluster_ICA_{}.csv'.format(dname))
	dbrp = pd.read_csv('./results/cluster_RP_{}.csv'.format(dname))
	dbrf = pd.read_csv('./results/cluster_RF_{}.csv'.format(dname))
	db = pd.read_csv('./results/cluster_{}.csv'.format(dname))
		
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))

	# plot SSE for K-Means
	metric = db['SSE_km']
	ax1.plot(db['k'], db['SSE_km'], marker='o', markersize=5, color='r', label='Original')
	ax1.plot(dbpca['k'], dbpca['SSE_km'], marker='o', markersize=5, color='c', label='PCA')
	ax1.plot(dbica['k'], dbica['SSE_km'], marker='o', markersize=5, color='m', label='ICA')
	ax1.plot(dbrp['k'], dbrp['SSE_km'], marker='o', markersize=5, color='b', label='RP')
	ax1.plot(dbrf['k'], dbrf['SSE_km'], marker='o', markersize=5, color='g', label='RF')
	ax1.set_title('K-Means SSE ({})'.format(dname))
	ax1.set_ylabel('Sum of squared error')
	ax1.set_xlabel('Number of clusters (k)')
	ax1.legend(loc='best', prop={'size': 6})
	ax1.grid(color='grey', linestyle='dotted')

	# plot Silhoutte Score for K-Means
	ax2.plot(db['k'], db['silhou_km'], marker='o', markersize=5, color='r', label='Original')
	ax2.plot(dbpca['k'], dbpca['silhou_km'], marker='o', markersize=5, color='c', label='PCA')
	ax2.plot(dbica['k'], dbica['silhou_km'], marker='o', markersize=5, color='m', label='ICA')
	ax2.plot(dbrp['k'], dbrp['silhou_km'], marker='o', markersize=5, color='b', label='RP')
	ax2.plot(dbrf['k'], dbrf['silhou_km'], marker='o', markersize=5, color='g', label='RF')
	ax2.set_title('K-Means Avg Silhouette Score ({})'.format(dname))
	ax2.set_ylabel('Mean silhouette score')
	ax2.set_xlabel('Number of clusters (k)')
	ax2.legend(loc='best', prop={'size': 6})
	ax2.grid(color='grey', linestyle='dotted')

	# plot log-likelihood for EM
	ax3.plot(db['k'], db['ll_em'], marker='o', markersize=5, color='r', label='Original')
	ax3.plot(dbpca['k'], dbpca['ll_em'], marker='o', markersize=5, color='c', label='PCA')
	ax3.plot(dbica['k'], dbica['ll_em'], marker='o', markersize=5, color='m', label='ICA')
	ax3.plot(dbrp['k'], dbrp['ll_em'], marker='o', markersize=5, color='b', label='RP')
	ax3.plot(dbrf['k'], dbrf['ll_em'], marker='o', markersize=5, color='g', label='RF')
	ax3.set_title('GMM Log-likelihood ({})'.format(dname))
	ax3.set_ylabel('Log-likelihood')
	ax3.set_xlabel('Number of clusters (k)')
	ax3.legend(loc='best', prop={'size': 6})
	ax3.grid(color='grey', linestyle='dotted')

	# plot BIC for EM
	ax4.plot(db['k'], db['bic_em'], marker='o', markersize=5, color='r', label='Original')
	ax4.plot(dbpca['k'], dbpca['bic_em'], marker='o', markersize=5, color='c', label='PCA')
	ax4.plot(dbica['k'], dbica['bic_em'], marker='o', markersize=5, color='m', label='ICA')
	ax4.plot(dbrp['k'], dbrp['bic_em'], marker='o', markersize=5, color='b', label='RP')
	ax4.plot(dbrf['k'], dbrf['bic_em'], marker='o', markersize=5, color='g', label='RF')
	ax4.set_title('GMM BIC ({})'.format(dname))
	ax4.set_ylabel('BIC')
	ax4.set_xlabel('Number of clusters (k)')
	ax4.legend(loc='best', prop={'size': 6})
	ax4.grid(color='grey', linestyle='dotted')

	# change layout size, font size and width between subplots
	fig.tight_layout()
	for ax in fig.axes:
		ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
		for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
			item.set_fontsize(8)
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('./plots/ReClusteringCombined_{}.png'.format(dname))

def plot_recluster_time(dname):
	dbpca = pd.read_csv('./results/cluster_PCA_{}.csv'.format(dname))
	dbica = pd.read_csv('./results/cluster_ICA_{}.csv'.format(dname))
	dbrp = pd.read_csv('./results/cluster_RP_{}.csv'.format(dname))
	dbrf = pd.read_csv('./results/cluster_RF_{}.csv'.format(dname))
	db = pd.read_csv('./results/cluster_{}.csv'.format(dname))
		
	fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

	# plot SSE for K-Means
	ax1.semilogy(db['k'], db['clk_time'], color='r', label='Original')
	ax1.semilogy(dbpca['k'], dbpca['clk_time'], color='c', label='PCA')
	ax1.semilogy(dbica['k'], dbica['clk_time'], color='m', label='ICA')
	ax1.semilogy(dbrp['k'], dbrp['clk_time'], color='b', label='RP')
	ax1.semilogy(dbrf['k'], dbrf['clk_time'], color='g', label='RF')
	ax1.set_title('K-Means Algorithm Time ({})'.format(dname))
	ax1.set_ylabel('Time (sec)')
	ax1.set_xlabel('Number of clusters (k)')
	ax1.legend(loc='best', prop={'size': 6})
	ax1.grid(color='grey', linestyle='dotted')


	# change layout size, font size and width between subplots
	fig.tight_layout()
	for ax in fig.axes:
		ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
		for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
			item.set_fontsize(8)
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('./plots/TimeClusteringCombined_{}.png'.format(dname))


if __name__ == "__main__":
	plot_recluster('Visa')
	plot_recluster('Letter')

	plot_recluster_time('Visa')
	plot_recluster_time('Letter')



