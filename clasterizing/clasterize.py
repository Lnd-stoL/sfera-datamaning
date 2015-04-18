
#-----------------------------------------------------------------------------------------------------------------------
# tunables

PLOT_IRIS_SAMPLE = False
PLOT_OPTIMAL_K   = False

CLUSTER_DISTANCE = 'average'
CLUSTER_COUNT    = 7
X_SLICE          = 8000
#SAVED_CLUSTERING = "lists/clusterized-6-average-8000.csv"
SAVED_CLUSTERING = "lists/clusterized-7-complete_linkage.csv"
#SAVED_CLUSTERING = "lists/clusterized-4-single_linkage.csv"

#-----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import pylab as pl
import numpy as np
import scipy.spatial as ss
import sklearn.cluster as sc
import sklearn.manifold as sm
import sklearn.datasets as ds
import sklearn.metrics as smt
import random
import math
import copy

from multiprocessing import Pool

#-----------------------------------------------------------------------------------------------------------------------

print "Loading features data frame ..."
data_df = pd.read_csv("./lists/features.csv", sep="\t", header=0, index_col="user_id")
x = data_df.values

#-----------------------------------------------------------------------------------------------------------------------

def objects_distance(o1, o2):
    cSumm = 0
    for i in xrange(len(o1)):
        cSumm += (o1[i] - o2[i]) ** 2;

    result = math.sqrt(float(cSumm))
    return result


class HierarchicalClustering:

    def __init__(self, K, Krange = [], **kwargs):
        self.K = K
        self.Krange = Krange
        self.cluster_dist_function = self.dist_complete_linkage
        self.clusters = {}
        return


    def cached_obj_distance(self, obj_id1, obj_id2):
        if obj_id1 < obj_id2: return self.dist_cache[obj_id1, obj_id2]
        return self.dist_cache[obj_id2, obj_id1]


    def set_cluster_distance_func(self, funcName):
        if funcName == 'average': self.cluster_dist_function = self.dist_average
        if funcName == 'single_linkage': self.cluster_dist_function = self.dist_single_linkage
        if funcName == 'complete_linkage': self.cluster_dist_function = self.dist_complete_linkage


    def dist_single_linkage(self, cluster1, cluster2, hint):
        minDist = self.cached_obj_distance(cluster1[0], cluster2[0])

        for objId1 in cluster1:
            for objId2 in cluster2:
                dist = self.cached_obj_distance(objId1, objId2)
                if dist < minDist:
                    minDist = dist

        return minDist


    def dist_complete_linkage(self, cluster1, cluster2, hint):
        maxDist = self.cached_obj_distance(cluster1[0], cluster2[0])

        for objId1 in cluster1:
            for objId2 in cluster2:
                #if hint != 0 and maxDist > hint: return -1.0
                dist = self.cached_obj_distance(objId1, objId2)
                if dist > maxDist:
                    maxDist = dist

        return maxDist


    def dist_average(self, cluster1, cluster2, hint):
        avgDist = 0
        elCount = len(cluster1) * len(cluster2)

        for objId1 in cluster1:
            for objId2 in cluster2:
                #if hint != 0 and avgDist / elCount  > hint: return -1.0
                avgDist += self.cached_obj_distance(objId1, objId2)

        return avgDist / elCount


    def precalc_objects_distances(self):
        print "\tcalculating distances caches ..."
        self.dist_cache = ss.distance.squareform(ss.distance.pdist(self.x))

        print "\tcalculating minimums ..."
        for i in xrange(self.x_len):
            rowMin = 10
            for j in xrange(self.x_len):
                if i == j: continue
                if self.dist_cache[i, j] < rowMin: rowMin = self.dist_cache[i, j]
            self.dist_cache[i, i] = rowMin


    def fit(self, x, y=None):
        self.x_len = len(x)
        return self


    def assign_clusters_to_x(self, clusters):
        print "\tassigning clusters to x elements ..."
        clustered_x = np.ndarray(len(self.x))
        for i in xrange(len(self.x)):
            for c in xrange(len(clusters)):
                if i in clusters[c][0]:
                    clustered_x[i] = c
                    break
        return clustered_x


    def cached_cluster_dist(self, i1, i2):
        if i1 > i2: return self.dist_cache[i1, i2]
        return self.dist_cache[i2, i1]


    def cache_cluster_dist(self, i1, i2, dist):
        if i1 > i2: self.dist_cache[i1, i2] = dist
        else:       self.dist_cache[i2, i1] = dist


    def predict(self, x):
        self.x = x
        self.precalc_objects_distances()
        clusters = [([i], i) for i in range(len(x))]

        secondMergeCandidates = clusters[0], clusters[1]
        secondDistance = self.cached_cluster_dist(secondMergeCandidates[0][1], secondMergeCandidates[1][1])

        while len(clusters) > self.K:
            if len(clusters) in self.Krange:
                self.clusters[len(clusters)] = copy.deepcopy(clusters)
            print "\tagglomerating clusters ", len(clusters)

            mergeCandidates = secondMergeCandidates
            minDistance = secondDistance

            skips = 0
            for i in xrange(len(clusters)):
                i_cid = clusters[i][1]
                if self.dist_cache[i_cid, i_cid] >= minDistance: continue

                for j in xrange(i+1, len(clusters)):
                    j_cid = clusters[j][1]
                    if self.dist_cache[j_cid, j_cid] >= minDistance: continue

                    dist = self.cached_cluster_dist(clusters[i][1], clusters[j][1])
                    if dist < secondDistance and dist >= minDistance:
                        secondDistance = dist
                        secondMergeCandidates = clusters[i], clusters[j]


                    if dist < minDistance:
                        secondDistance = minDistance
                        secondMergeCandidates = mergeCandidates
                        minDistance = dist
                        mergeCandidates = clusters[i], clusters[j]

            print "\tmerging ..."
            mergeCandidates[0][0].extend(mergeCandidates[1][0])
            for i in xrange(len(clusters)):
                ind1, ind2 = mergeCandidates[0][1], clusters[i][1]
                d = self.cluster_dist_function(mergeCandidates[0][0], clusters[i][0], 0)
                self.cache_cluster_dist(ind1, ind2, d)
                if d < self.dist_cache[ind1, ind1]: self.dist_cache[ind1, ind1] = d
                if d < self.dist_cache[ind2, ind2]: self.dist_cache[ind2, ind2] = d

            if secondMergeCandidates[0][1] == mergeCandidates[0][1] or secondMergeCandidates[1][1] == mergeCandidates[1][1] or secondMergeCandidates[0][1] == mergeCandidates[1][1] or secondMergeCandidates[1][1] == mergeCandidates[0][1]:
                secondMergeCandidates = (0, 0), (0, 0)
                secondDistance = float("+inf")

            for c in xrange(len(clusters)):
                if clusters[c][1] == mergeCandidates[1][1]:
                    del clusters[c]
                    break

        return self.assign_clusters_to_x(clusters)


    def hierarhy_level(self, k):
        return self.assign_clusters_to_x(self.clusters[k])


    def fit_predict(self, x, y=None):
        self.fit(x, y)
        return self.predict(x)

#-----------------------------------------------------------------------------------------------------------------------

iris = ds.load_iris()
x_iris = iris.data[:100]
y_iris = iris.target[:100]

if PLOT_IRIS_SAMPLE:
    pl.figure(figsize=(10, 5))

    pl.subplot(1, 2, 1)
    pl.scatter(x_iris[:, 0], x_iris[:, 1], c=y_iris, cmap=pl.cm.PuOr, lw=0, s=30)
    pl.xlabel('Sepal length')
    pl.ylabel('Sepal width')

    pl.subplot(1, 2, 2)
    pl.scatter(x_iris[:, 2], x_iris[:, 3], c=y_iris, cmap=pl.cm.PuOr, lw=0, s=30)
    pl.xlabel('Petal length')
    pl.ylabel('Petal width')
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------

def quality_diameter(x, y, K, distance):
    clusters = []
    for i in xrange(k):
        clusters.append([])

    for sample, i in zip(x, y):
        clusters[i].append(sample)
    sum_diameter = 0.0

    for cluster in clusters:
        diameter = 0.0
        for i in xrange(len(cluster)):
            for j in xrange(i + 1, len(cluster)):
                if diameter < distance[i][j]:
                    diameter = distance[i][j]
        sum_diameter += diameter

    return sum_diameter / len(clusters)


def quality(x, y, K, d_cache):    # silhouette
    print "\tcalculating silhouette ..."
    result = 0.0

    for i in xrange(len(x)):
        ai = 0.0
        bi = np.zeros(len(x), dtype=float)
        na = 0
        nb = np.zeros(len(x), dtype=int)

        for j in xrange(len(x)):
            if y[j] == y[i]:
                na += 1
                ai += d_cache[i, j]
            else:
                nb[y[j]] += 1
                bi[y[j]] += d_cache[i, j]

        ai /= na
        minBi = bi[0]
        for j in xrange(K):
            if j == y[i]: continue;
            bi[j] /= float(nb[j])
            if bi[j] != 0 and bi[j] < minBi: minBi = bi[j]

        result += (minBi - ai) / max(minBi, ai)

    return result / float(len(x))

#-----------------------------------------------------------------------------------------------------------------------

clustering = HierarchicalClustering(2)
clustering.set_cluster_distance_func(CLUSTER_DISTANCE)
pred_iris = clustering.fit_predict(x_iris)
print "Adjusted Rand index for iris is: %.2f" % smt.adjusted_rand_score(y_iris, pred_iris)
#print "Silhouette for iris is: %.2f" % quality(x_iris, y_iris, self.dist_cache)


#-----------------------------------------------------------------------------------------------------------------------
# find the best cluster count (manually using elbow)

if SAVED_CLUSTERING is None:
    x = x[:X_SLICE]
    x_pairwise_dist_cache = ss.distance.squareform(ss.distance.pdist(x))

    ks = range(3, 12)
    criteria = np.zeros(len(ks))

    cls = HierarchicalClustering(ks[0]-1, list(ks))
    cls.set_cluster_distance_func(CLUSTER_DISTANCE)
    y = cls.fit_predict(x)


    if PLOT_OPTIMAL_K:
        for i, k in enumerate(ks):
            y = cls.hierarhy_level(k)
            criteria[i] = quality(x, y, k, x_pairwise_dist_cache)

        pl.figure(figsize=(8, 6))
        pl.plot(ks, criteria)
        pl.title("$J(k)$")
        pl.ylabel("Criteria $J$")
        pl.xlabel("Number of clusters $k$")
        pl.grid()
        pl.show()

#-----------------------------------------------------------------------------------------------------------------------
# clasterize twitter users now

    k = CLUSTER_COUNT
    y = cls.hierarhy_level(k)

    df_out = pd.DataFrame(data=x, columns=data_df.columns)
    df_out["cluster"] = y
    df_out.to_csv("./lists/clusterized-" + str(CLUSTER_COUNT) + "-" + str(CLUSTER_DISTANCE) + "-" + str(X_SLICE) + ".csv", sep="\t")

#-----------------------------------------------------------------------------------------------------------------------
else:
    df_in = pd.read_csv(SAVED_CLUSTERING, sep="\t", header=0)
    x = df_in[data_df.columns].values
    y = df_in["cluster"].values;
    k = CLUSTER_COUNT


print "Calculating t-SNE ..."
tsne = sm.TSNE(n_components=2, verbose=1, n_iter=1000)
z = tsne.fit_transform(x)

# Color map
cm = pl.get_cmap('hot')
pl.figure(figsize=(15, 15))
pl.scatter(z[:, 0], z[:, 1], c=map(lambda c: cm(1.0 * c / k), y))
pl.axis('off')
pl.show()

#-----------------------------------------------------------------------------------------------------------------------

print "Plotting radar ..."

def radar(centroid, features, axes, color):
    # Set ticks to the number of features (in radians)
    t = np.arange(0, 2*np.pi, 2*np.pi/len(features))
    plt.xticks(t, [])

    # Set yticks from 0 to 1
    plt.yticks(np.linspace(0, 1, 6))

    # Draw polygon representing centroid
    points = [(x, y) for x, y in zip(t, centroid)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO,] + [path.Path.LINETO,] * (len(centroid) - 1) + [ path.Path.CLOSEPOLY ]
    _path = path.Path(points, codes)
    _patch = patches.PathPatch(_path, fill=True, color=color, linewidth=0, alpha=.3)
    axes.add_patch(_patch)
    _patch = patches.PathPatch(_path, fill=False, linewidth = 2)
    axes.add_patch(_patch)

    # Draw circles at value points
    plt.scatter(points[:,0], points[:,1], linewidth=2, s=50, color='white', edgecolor='black', zorder=10)

    # Set axes limits
    plt.ylim(0, 1)

    # Draw ytick labels to make sure they fit properly
    for i in range(len(features)):
        angle_rad = i/float(len(features))*2*np.pi
        angle_deg = i/float(len(features))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 1.05, features[i], size=7, horizontalalignment=ha, verticalalignment="center")

# Some additiola imports
import matplotlib
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Choose some nice colors
matplotlib.rc('axes', facecolor = 'white')
# Make figure background the same colors as axes
fig = plt.figure(figsize=(15, 15), facecolor='white')

cm = pl.get_cmap('jet')

clusters = np.unique(y)
for j, cluster in enumerate(clusters):
    x_c = x[y == cluster]
    centroid = x_c.mean(axis=0)
    # Use a polar axes
    axes = plt.subplot(3, 3, j + 1, polar=True)
    radar(centroid, data_df.columns.values, axes, cm(1.0 * j / k))

plt.show()
