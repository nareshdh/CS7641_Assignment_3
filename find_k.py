import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from helpers import *

perm_x, perm_y, letter_x, letter_y = load_data()
K = [2,3,4,5,7,10,15,20,25,30,35,40,45,50,55,60]
KM = [KMeans(n_clusters=k).fit(letter_x) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(letter_x, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/letter_x.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(letter_x)**2)/letter_x.shape[0]
bss = tss-wcss

seg_threshold = 0.95 #Set this to your desired target

#The angle between three points
def segments_gain(p1, v, p2):
    vp1 = np.linalg.norm(p1 - v)
    vp2 = np.linalg.norm(p2 - v)
    p1p2 = np.linalg.norm(p1 - p2)
    return np.arccos((vp1**2 + vp2**2 - p1p2**2) / (2 * vp1 * vp2)) / np.pi

#Normalize the data
criterion = np.array(avgWithinSS)
criterion = (criterion - criterion.min()) / (criterion.max() - criterion.min())

#Compute the angles
seg_gains = np.array([0, ] + [segments_gain(*[np.array([K[j], criterion[j]]) for j in range(i-1, i+2)]) for i in range(len(K) - 2)] + [np.nan, ])
#import pdb
#pdb.set_trace()
#Get the first index satisfying the threshold
kIdx = np.argmax(seg_gains > seg_threshold)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.show()