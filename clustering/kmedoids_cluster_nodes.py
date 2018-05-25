"""!

@brief Filip Hörnsten's Bachelor degree project 2018

@authors Filip Hörnsten
@date 2018
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond
"""

from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

from pyclustering.utils import calculate_distance_matrix, average_inter_cluster_distance, average_intra_cluster_distance, read_sample, timedcall;
from pyclustering.utils.metric import distance_metric, type_metric;

from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.kmedoids import kmedoids;
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from silhouette_value import calculate_silhouette
from operator import itemgetter

import os;
import numpy;
import random;
import time

"""
def silhouette_value(clusters, sample):
    silhouette = []
    #print("DEBUG:", clusters[0])
    for index in range(len(clusters)):
        if(index < len(clusters)-1):
            #print("DEBUG2:", index)
            inter_clust_dist = average_inter_cluster_distance(clusters[index], clusters[index+1], data = sample)
            intra_clust_dist = average_intra_cluster_distance(clusters[index], clusters[index+1], data = sample)

            #print("inter:", inter_clust_dist)
            #print("intra:", intra_clust_dist)

            # Calculate the silhouette value
            #print("maximum:", numpy.maximum(intra_clust_dist, inter_clust_dist))
            silhouette.append(((inter_clust_dist - intra_clust_dist) / numpy.maximum(intra_clust_dist, inter_clust_dist)))
            #print("silhouette:", silhouette)

    return numpy.mean(silhouette)
"""

def cluster_nodes(visualisation=False):
    # (kmedoids_cluster_nodes.py
    # template_clustering()

    # K-medoids clustering using points as data
    
    # Get the nodes sample from the data file
    # samplePath = os.path.dirname(os.path.abspath("kmedoids_cluster_nodes.py")) + os.sep + "nodes-test1.data"
    
    #scenarios = ["scenario1.data", "scenario2.data", "scenario3.data", "scenario4.data", "scenario5.data"]
    scenarios = ["scenario1.data", "scenario2.data", "scenario3.data", "scenario4.data", "scenario5.data"]
    """
    #random.seed(35344796)
    random.seed(35334096)
    scenario4 = []
    for i in range(10):
        scenario4.append(random.randrange(1000))
    
    print(scenario4)
    """
    
    # [564, 162, 959, 271, 663, 992, 566, 883, 438, 118] # 1k
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17, 25], [8, 12, 22, 28]];
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25], [103, 16, 196, 76, 80, 41, 47, 52, 112, 199], [10461, 7157, 11717, 2709, 13116, 16811, 2041, 19481, 9130, 12817]];
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25], [103, 16, 196, 76, 80, 41, 47, 52, 112, 199], [8257, 3356, 10812, 2440, 14783, 10547, 11063, 11980, 6929, 18896]]
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25], [103, 16, 196, 76, 80, 41, 47, 52, 112, 199], [1129, 324, 1919, 542, 1326, 1985, 1133, 1767, 877, 237], [103, 16, 196, 76, 80]]
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25], [103, 16, 196, 76, 80, 41, 47, 52, 112, 199], [564, 162, 959, 271, 663, 992, 566, 883, 438, 118]]
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25], [103, 16, 196, 76, 80, 41, 47, 52, 112, 199]];
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [15, 25, 28, 32, 10]]; 0.2476339234441543
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [13, 23, 28, 32, 8]];

    scenarioClustersDistanceList = []

    for scenarioIndex in range(0, len(scenarios)):
        total_time_start = time.perf_counter()
        total_wall_time_start = time.time()
        samplePath = os.path.dirname(os.path.abspath("kmedoids_cluster_nodes.py")) + os.sep + scenarios[scenarioIndex]
        sample = read_sample(samplePath)

        print("\nScenario", scenarioIndex+1, "\nSample:", samplePath, "\n")

        # Use Manhattan distance
        metric = distance_metric(type_metric.MANHATTAN);

        # Store the silhouette value for different number of clusters (k)
        silhouettes = []
        
        # Run clustering k times, calculate silhouette value for each time and choose clustering with best value
        for k in range(2, 11):
            # Randomly generate the medoids
            """random.seed(35334096)
            random_medoids = []
            for i in range(k):
                random_medoids.append(random.randrange(len(sample)))
"""
            # Initialise the clustering algorithm with the k-means++ algorithm
            # Initialise the random generator with a seed for reproducibility
            random.seed(35334096)
            initial_points = kmeans_plusplus_initializer(sample, k).initialize()
            #print(sample)
            #print("Type of initial points:", type(initial_points))
            #print("Initial points:", initial_points)
            initial_medoids = []
            for point in sample:
                #print("Sample point:", point)
                for initial_point in initial_points:
                    #print("Single point type:", type(initial_point))
                    if(point[0] == initial_point[0] and point[1] == initial_point[1]):
                        initial_medoids.append(sample.index(point))
            #print("Initial medoids:", initial_medoids)
            

            #print("Random medoids:", random_medoids)
            
            # Initiate the k-medoids algorithm with the sample and the initial medoids
            #kmedoids_instance = kmedoids(sample, initial_medoids[scenarioIndex], 0.001, metric=metric, ccore = True);
            kmedoids_instance = kmedoids(sample, initial_medoids, 0.001, metric=metric, ccore = True);

            # Start performance counter
            time_start = time.perf_counter()
            wall_time_start = time.time()

            # Perform actual clustering
            kmedoids_instance.process()

            # Stop performance counter
            time_end = time.perf_counter()
            wall_time_end = time.time()

            # Calculate execution time and wall time
            clustering_time = time_end - time_start
            clustering_wall_time = wall_time_end - wall_time_start
            print("Execution time for clustering for k=" + str(k) + ":", clustering_time, "\nWall time for clustering for k=" + str(k) + ":", clustering_wall_time)
            
            # by default k-medoids returns representation CLUSTER_INDEX_LIST_SEPARATION
            clusters = kmedoids_instance.get_clusters()
            medoids = kmedoids_instance.get_medoids();
            #print("Clusters before changing encoding:", clusters)
            type_repr = kmedoids_instance.get_cluster_encoding();
            #print("Representator type:", type_repr)
            
            encoder = cluster_encoder(type_repr, clusters, sample);
        
            # change representation from index list to label list
            encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING);
            #kmedoids_instance.process
            type_repr2 = encoder.get_encoding;

            # Cluster representation converted from a list of sample indexes to their respective labels
            cluster_labels = encoder.get_clusters()
            
            #print("Representator type afterwards:", type_repr2)
            #print("Cluster labels", cluster_labels)

            #print("Number of medoids:", len(medoids))
            #[[float(y) for y in x] for x in l]
            medoidPoints = [[point for point in sample[index]] for index in medoids]
            #print("Medoid points:", medoidPoints)

            # Calculate the silhouette value
            silhouettes.append((calculate_silhouette(sample, cluster_labels, medoidPoints, k, visualisation), k, clustering_time, clustering_wall_time, clusters, medoids))
            
            # Calculate Silhouette value
            #sil_val = silhouette_value(kmedoids_instance.get_clusters(), sample)
            #print("pyclustering silhouette value for", k, "clusters:", sil_val)
        
        best_silhouette, best_k, best_time, best_wall_time, best_clusters, best_medoids = max(silhouettes,key=itemgetter(0))
        print("The best silhouette value of", best_silhouette, "was achieved with k=" + str(best_k) + "\nExecution time of best clustering: " + str(best_time) + "\nWall time of best clustering: " + str(best_wall_time))
        #print("Best clusters:", best_clusters)
        #print("Best medoids:", best_medoids)

        # Run clustering and print result of clustering as well as execution time
        #(ticks, result) = timedcall(kmedoids_instance.process);
        #print( "\nExecution time:", time);
        #clusters = kmedoids_instance.get_clusters();
        #medoids = kmedoids_instance.get_medoids();
        #print("Clusters:", clusters);
        #print("Medoids:", medoids)

        # Generate visualisation
        if(visualisation):
            title = "K-medoids clustering - Scenario " + str(scenarioIndex+1)
            visualizer = cluster_visualizer(1, titles=[title]);
            visualizer.append_clusters(best_clusters, sample, 0);
            #visualizer.append_cluster([ sample[index] for index in initial_medoids[index] ], marker = '*', markersize = 15);
            visualizer.append_cluster(best_medoids, data=sample, marker='*', markersize=15, color="black");
            visualizer.show(visible_axis = False, visible_grid = False);

        # Post-processing
        
        # Calculate Manhattan distance from medoid to all points in the cluster
        metric = distance_metric(type_metric.MANHATTAN);
        clusterList = []
        #print("Number of clusters:", len(best_clusters),)
        for index in range(0, len(best_clusters)):
            #print("Index: ", index)
            medoidPoint = sample[best_medoids[index]]
            #print("Medoid point array: ", medoidPoint)
            #print("Cluster index array: ", clusters[index])
            nodeList = []
            for currentClusterIndex in best_clusters[index]:
                # Make sure not to compare the medoid to itself
                if best_medoids[index] != currentClusterIndex:
                    # Get the point array of the current cluster to compare to the medoid
                    currentClusterPoint = sample[currentClusterIndex]
                    #print("Current cluster point from sample:", currentClusterPoint)
                    
                    # Calculate the Manhattan distance between the medoid and the current point to compare with
                    distance = metric(medoidPoint, currentClusterPoint)

                    # Append the result to a list as the index of the medoid, the index of the current point and the distance between them
                    nodeList.append([best_medoids[index], currentClusterIndex, distance])
                    #print("Distance between ", medoidPoint, " and ", currentClusterPoint, " is: ", distance)
                    
            clusterList.append(nodeList)
            
        scenarioClustersDistanceList.append(clusterList)

        total_time_end = time.perf_counter()
        total_wall_time_end = time.time()
        print("\nTotal scenario execution time:", total_time_end - total_time_start, "\nTotal scenario wall time:", total_wall_time_end - total_wall_time_start, "\n\n----")
    
    return scenarioClustersDistanceList
    
    """# K-medoids clustering using distance matrix

    # calculate distance matrix for sample
    matrix = calculate_distance_matrix(sample);

    # create K-Medoids algorithm for processing distance matrix instead of points
    kmedoids_instance2 = kmedoids(matrix, initial_medoids, data_type='distance_matrix');

    # run cluster analysis and obtain results
    kmedoids_instance2.process();

    clusters2 = kmedoids_instance2.get_clusters();
    medoids2 = kmedoids_instance2.get_medoids();
    print("Clusters2: ", clusters);
    print("Medoids2: ", medoids)
    print("Distance matrix: ", matrix)

    # Generate visualisation
    visualizer = cluster_visualizer(1, titles=["K-medoids clustering of nodes using distance matrix"]);
    visualizer.append_clusters(clusters, sample, 0);
    visualizer.append_cluster([ sample[index] for index in initial_medoids ], marker = '*', markersize = 15);
    visualizer.append_cluster(medoids, data=sample, marker='*', markersize=15, color="black");
    visualizer.show(visible_axis = False, visible_grid = False);"""
    
"""def cluster_sample1():
    template_clustering([2, 9], SIMPLE_SAMPLES.SAMPLE_SIMPLE1);
    
def cluster_sample2():
    template_clustering([3, 12, 20], SIMPLE_SAMPLES.SAMPLE_SIMPLE2);
    
def cluster_sample3():
    template_clustering([4, 12, 25, 37], SIMPLE_SAMPLES.SAMPLE_SIMPLE3);
    
def cluster_sample4():
    template_clustering([4, 15, 30, 40, 50], SIMPLE_SAMPLES.SAMPLE_SIMPLE4);

def cluster_sample5():
    template_clustering([4, 18, 34, 55], SIMPLE_SAMPLES.SAMPLE_SIMPLE5);
        
def cluster_elongate():
    template_clustering([8, 56], SIMPLE_SAMPLES.SAMPLE_ELONGATE);

def cluster_lsun():
    template_clustering([10, 275, 385], FCPS_SAMPLES.SAMPLE_LSUN);

def cluster_target():
    template_clustering([10, 160, 310, 460, 560, 700], FCPS_SAMPLES.SAMPLE_TARGET);

def cluster_two_diamonds():
    template_clustering([10, 650], FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS);

def cluster_wing_nut():
    template_clustering([19, 823], FCPS_SAMPLES.SAMPLE_WING_NUT);
    
def cluster_chainlink():
    template_clustering([30, 900], FCPS_SAMPLES.SAMPLE_CHAINLINK);
    
def cluster_hepta():
    template_clustering([0, 35, 86, 93, 125, 171, 194], FCPS_SAMPLES.SAMPLE_HEPTA);
    
def cluster_tetra():
    template_clustering([0, 131, 214, 265], FCPS_SAMPLES.SAMPLE_TETRA);

def cluster_atom():
    template_clustering([0, 650], FCPS_SAMPLES.SAMPLE_ATOM);

def cluster_engy_time():
    template_clustering([10, 3000], FCPS_SAMPLES.SAMPLE_ENGY_TIME);


def display_fcps_clustering_results():
    (lsun, lsun_clusters) = template_clustering([10, 275, 385], FCPS_SAMPLES.SAMPLE_LSUN, 0.1, False);
    (target, target_clusters) = template_clustering([10, 160, 310, 460, 560, 700], FCPS_SAMPLES.SAMPLE_TARGET, 0.1, False);
    (two_diamonds, two_diamonds_clusters) = template_clustering([10, 650], FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, 0.1, False);
    (wing_nut, wing_nut_clusters) = template_clustering([19, 823], FCPS_SAMPLES.SAMPLE_WING_NUT, 0.1, False);
    (chainlink, chainlink_clusters) = template_clustering([30, 900], FCPS_SAMPLES.SAMPLE_CHAINLINK, 0.1, False);
    (hepta, hepta_clusters) = template_clustering([0, 35, 86, 93, 125, 171, 194], FCPS_SAMPLES.SAMPLE_HEPTA, 0.1, False);
    (tetra, tetra_clusters) = template_clustering([0, 131, 214, 265], FCPS_SAMPLES.SAMPLE_TETRA, 0.1, False);
    (atom, atom_clusters) = template_clustering([0, 650], FCPS_SAMPLES.SAMPLE_ATOM, 0.1, False);
    
    visualizer = cluster_visualizer(8, 4);
    visualizer.append_clusters(lsun_clusters, lsun, 0);
    visualizer.append_clusters(target_clusters, target, 1);
    visualizer.append_clusters(two_diamonds_clusters, two_diamonds, 2);
    visualizer.append_clusters(wing_nut_clusters, wing_nut, 3);
    visualizer.append_clusters(chainlink_clusters, chainlink, 4);
    visualizer.append_clusters(hepta_clusters, hepta, 5);
    visualizer.append_clusters(tetra_clusters, tetra, 6);
    visualizer.append_clusters(atom_clusters, atom, 7);
    visualizer.show();"""


"""cluster_sample1();
cluster_sample2();
cluster_sample3();
cluster_sample4();
cluster_sample5();
cluster_elongate();
cluster_lsun();
cluster_target();
cluster_two_diamonds();
cluster_wing_nut();
cluster_chainlink();
cluster_hepta();
cluster_tetra();
cluster_atom();
cluster_engy_time();"""


# display_fcps_clustering_results();
