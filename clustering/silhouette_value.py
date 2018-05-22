from __future__ import print_function

"""

Filip Hörnsten

http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from nipy import nipy_spectral
import numpy as np

#print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.

def controlgroup():
    X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility
    for k in range(2,7):
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        visualise_silhouette(X, cluster_labels, clusterer.cluster_centers_, k)


#range_n_clusters = [2, 3, 4, 5, 6]

def calculate_silhouette(sample, cluster_labels, medoids, k, visualisation=False):
    # Input pre-processing / sanity checking
    if(not isinstance(sample, np.ndarray)):
        sample = np.asarray(sample)
        #print("Converted sample:", sample)
        #print("Converted type of sample:", type(sample))

    if(not isinstance(cluster_labels, np.ndarray)):
        cluster_labels = np.asarray(cluster_labels)
        #print("Converted cluster labels:", cluster_labels)
        #print("Converted type of cluster labels:", type(cluster_labels))

    if(not isinstance(medoids, np.ndarray)):
        medoids = np.asarray(medoids)
        #print("Converted medoids:", medoids)
        #print("Converted type of medoids:", type(medoids))
        
    #print("Sample:", sample)
    #print("Labels:", cluster_labels)
    #print("Medoids:", medoids)

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    #clusterer = KMeans(n_clusters=k, random_state=10)
    #cluster_labels = clusterer.fit_predict(X) #OLD

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    #print("X:", X)
    #print("cluster labels:", cluster_labels)
    silhouette_avg = silhouette_score(sample, cluster_labels)
    print("For k =", k,
          "The average silhouette_score is :", silhouette_avg)

    if(visualisation):
        #for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.5, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(sample) + (k + 1) * 10])
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(sample, cluster_labels)
        #print("Silhouette sample values:", sample_silhouette_values)

        y_lower = 10
        #print("K:", k)
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            #print("i:", i)
            #print("Cluster labels == i:", cluster_labels == i)
            #print("Sample silhouette values that are true:", sample_silhouette_values[cluster_labels == i])
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / k)
            #print("Color:", color)
            #print("ith cluster silhouette value:", ith_cluster_silhouette_values)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        #colors = cm.spectral(cluster_labels.astype(float) / n_clusters) #OLD
        colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / k)
        #print("Type of sample:", type(sample))
        ax2.scatter(sample[:, 0], sample[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        #medoids = clusterer.cluster_centers_
        # Draw white circles at cluster medoids
        ax2.scatter(medoids[:, 0], medoids[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(medoids):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for K-medoids clustering on sample data "
                      "with k = %d" % k),
                     fontsize=14, fontweight='bold')

        plt.show()
    return silhouette_avg

#controlgroup()
#visualise_silhoutte(X, y)
