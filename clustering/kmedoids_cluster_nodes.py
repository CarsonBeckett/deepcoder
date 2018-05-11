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

from pyclustering.utils import calculate_distance_matrix;
from pyclustering.utils.metric import distance_metric, type_metric;

from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.kmedoids import kmedoids;

from pyclustering.utils import read_sample;
from pyclustering.utils import timedcall;

import os;

"""def template_clustering(start_medoids, path, tolerance = 0.25, show = True):
    sample = read_sample(path);
    
    kmedoids_instance = kmedoids(sample, start_medoids, tolerance);
    (ticks, result) = timedcall(kmedoids_instance.process);
    
    clusters = kmedoids_instance.get_clusters();
    medoids = kmedoids_instance.get_medoids();
    print("Sample: ", path, "\t\tExecution time: ", ticks, "\n");

    if show is True:
        visualizer = cluster_visualizer(1);
        visualizer.append_clusters(clusters, sample, 0);
        visualizer.append_cluster([ sample[index] for index in start_medoids ], marker = '*', markersize = 15);
        visualizer.append_cluster(medoids, data=sample, marker='*', markersize=15);
        visualizer.show();
    
    return (sample, clusters);"""

def cluster_nodes():
    # (kmedoids_cluster_nodes.py
    # template_clustering()

    # K-medoids clustering using points as data
    
    # Get the nodes sample from the data file
    # samplePath = os.path.dirname(os.path.abspath("kmedoids_cluster_nodes.py")) + os.sep + "nodes-test1.data"
    
    scenarios = ["scenario1.data", "scenario2.data", "scenario3.data"]
    #initial_medoids = [[8, 12, 17, 25], [8, 12, 17, 25], [8, 12, 22, 28]];
    initial_medoids = [[8, 12, 17, 25], [8, 12, 17], [9, 12, 25]];

    for index in range(0, len(scenarios)):
        samplePath = os.path.dirname(os.path.abspath("kmedoids_cluster_nodes.py")) + os.sep + scenarios[index]
        sample = read_sample(samplePath)
        print(samplePath)
        print(sample)

        # Use Manhattan distance
        metric = distance_metric(type_metric.MANHATTAN);
        
        # Initiate the k-medoids algorithm with the sample and the initial medoids
        kmedoids_instance = kmedoids(sample, initial_medoids[index], 0.001, metric=metric);

        # Run clustering and print results
        kmedoids_instance.process();
        clusters = kmedoids_instance.get_clusters();
        medoids = kmedoids_instance.get_medoids();
        print("Clusters: ", clusters);
        print("Medoids: ", medoids)

        # Generate visualisation
        title = "K-medoids clustering - Scenario " + str(index+1)
        visualizer = cluster_visualizer(1, titles=[title]);
        visualizer.append_clusters(clusters, sample, 0);
        #visualizer.append_cluster([ sample[index] for index in initial_medoids[index] ], marker = '*', markersize = 15);
        visualizer.append_cluster(medoids, data=sample, marker='*', markersize=15, color="black");
        visualizer.show(visible_axis = False, visible_grid = False);

        # Post-processing
        
        # Calculate Manhattan distance from medoid to all points in the cluster
        metric = distance_metric(type_metric.MANHATTAN);
        distanceList = []
        print("Number of clusters: ", len(clusters))
        for index in range(0, len(clusters)):
            print("Index: ", index)
            medoidPoint = sample[medoids[index]]
            print("Medoid point array: ", medoidPoint)
            print("Cluster index array: ", clusters[index])
            for currentClusterIndex in clusters[index]:
                # Make sure not to compare the medoid to itself
                if medoids[index] != currentClusterIndex:
                    # Get the point array of the current cluster to compare to the medoid
                    currentClusterPoint = sample[currentClusterIndex]
                    print("Current cluster point from sample:", currentClusterPoint)
                    
                    # Calculate the Manhattan distance between the medoid and the current point to compare with
                    distance = metric(medoidPoint, currentClusterPoint)

                    # Append the result to a list as the index of the medoid, the index of the current point and the distance between them
                    distanceList.append([medoids[index], currentClusterIndex, distance])
                    print("Distance between ", medoidPoint, " and ", currentClusterPoint, " is: ", distance)

        print(distanceList)

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

cluster_nodes()
