"""!

@brief Filip Hörnsten's Bachelor degree project 2018

@authors Filip Hörnsten
@date 2018
@copyright GNU Public License
"""

from operator import itemgetter
from kmedoids_cluster_nodes import cluster_nodes

# Assign a rank to every node in a cluster depending on its distance to the representative object of the cluster (in this case its medoid)
def rank(cluster):
    ranks = []
    print("Original: ", cluster)

    # Sort the cluster list in descending order on the distance to the representative object of the cluster
    # The lower the distance the higher the index/rank
    sortedCluster = sorted(cluster, key=itemgetter(2), reverse=True)
    print("Sorted: ", sortedCluster)

    # Build output list by appending the index/ID of the node and its priority
    for index in range(len(sortedCluster)):
        ranks.append([sortedCluster[index][1], index])
    
    return ranks

# Reference monarchical leader election that selects the process with the highest rank as the leader
# For the theory behind this simple algorithm, see p.53 in
# Cachin, C., Guerraoui, R., & Rodrigues, L. (2011). Introduction to reliable and secure distributed programming. 2nd edition. Springer Science & Business Media.
def monarchical_leader_election(ranking):
    return max(ranking)

#Run the program
scenarios = cluster_nodes()
clusterList = []
for scenario in scenarios:
    #print(clusters)
    #print(scenarios == clusters)
    for clusters in scenario:
        clusterList.append(clusters)
    
print("CLUSTERLIST: ", clusterList[0])
#print(scenarios)
leaders = []

print("Rank: ", rank(clusterList[0]))
leader = monarchical_leader_election(ranks)
print("Index:", leader, ", Node ID:", ranking[leader][0])


"""for cluster in clusterList:
    ranking = rank(cluster)
    print(ranking)
    ranks = []
    for rank in ranking:
        ranks.append(rank[1])
        
    leader = monarchical_leader_election(ranks)
    print("Index:", leader, ", Node ID:", ranking[leader][0])
"""
"""ranking = rank([[15, 11, 0.6663620000000003], [15, 12, 0.7182299999999993], [15, 13, 0.058244999999999436], [15, 14, 0.4529019999999999]])
print(ranking)
ranks = []
for rank in ranking:
    ranks.append(rank[1])
leader = monarchical_leader_election(ranks)
print("Index:", leader, ", Node ID:", ranking[leader][0])"""
