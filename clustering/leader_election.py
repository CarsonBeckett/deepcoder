"""!

@brief Filip Hörnsten's Bachelor degree project 2018

@authors Filip Hörnsten
@date 2018
@copyright GNU Public License
"""

# -----

import time

import numpy as np

import tqdm

from deepcoder import context, util
from deepcoder.dsl.value import IntValue, NULLVALUE
from deepcoder.dsl.function import OutputOutOfRangeError, NullInputError
from deepcoder.dsl.program import Program, get_unused_indices

# -----


import numpy
import time

from operator import itemgetter
from kmedoids_cluster_nodes import cluster_nodes
from deepcoder.scripts.solve_problems import solve_problem

from deepcoder import context
from deepcoder.search import dfs, sort_and_add
from deepcoder.dsl import impl
from deepcoder.dsl import types
from deepcoder.dsl.value import Value

# Assign a rank to every node in a cluster depending on its distance to the representative object of the cluster (in this case its medoid)
def rank(cluster):
    ranks = []
    #print("Original: ", cluster)

    # Sort the cluster list in descending order on the distance to the representative object of the cluster
    # The lower the distance the higher the index/rank
    sortedCluster = sorted(cluster, key=itemgetter(2), reverse=True)
    #print("Sorted: ", sortedCluster)

    # Build output list by appending the index/ID of the node and its priority
    for index in range(len(sortedCluster)):
        ranks.append([sortedCluster[index][1], index])
    
    return ranks

# Reference monarchical leader election that selects the process with the highest rank as the leader
# For the theory behind this simple algorithm, see p.53 in
# Cachin, C., Guerraoui, R., & Rodrigues, L. (2011). Introduction to reliable and secure distributed programming. 2nd edition. Springer Science & Business Media.
def monarchical_leader_election(ranking):
    return max(ranking)

# -----
"""
# Run the clustering algorithm
scenarios = cluster_nodes()

# Extract the clusters in each scenario from a list of all scenarios
clusterList = []
for scenario in scenarios:
    for clusters in scenario:
        clusterList.append(clusters)
    
print("CLUSTERLIST: ", clusterList[0])
#print(scenarios)
rankSets = rank(clusterList[0])
print("Rank sets: ", rankSets)
ranks = []
for i in range(len(rankSets)):
    ranks.append(rankSets[i][1])
print("Rank: ", ranks)


# Ground truth / oracle
leader = monarchical_leader_election(ranks)
print("Index:", leader, ", Node ID:", rankSets[leader][0])

# Pass rank array and elected leader as input-output examples to DeepCoder
# deepcoder_result = solve_problem([{rank, leader}],2)
#deepcoder_result = solve_problem("dataset/T=2_test.json",2)

#examples = [([ranks], leader), ([[0,1,2,3,4,5]], 5)]
examples = [([[5,1,8,3,4,0,6,7,2]], 8), ([[1,0,2,5,4,3]], 5)]
print("examples:", examples)
predictions = numpy.zeros(len(impl.FUNCTIONS))
print("predictions:", predictions)
scores = dict(zip(impl.FUNCTIONS, predictions))
print("scores:", scores)
ctx = context.Context(scores)
print("context:", ctx)

print("examples 00:", examples[0][0])
#input_types = [x.type for x in examples[0][0]]
#print("Input types:", input_types)
#Type(): <class 'list'>
#Type(): <class 'list'>

#Type(): <class 'list'>
#Type(): <class 'deepcoder.dsl.value.ListValue'>

# [([[13, -147, -30, 15, -110, -85, 66, -240, -111, 132, 236, -149, -76, -163, -159, -34, -225, 197, 26]], [13, -147, -30, 15, -110, -85, 66, -240, -111, 132, 236, -149, -76, -163, -159, -34, -225, 197, 26]), ([[-118, -81, 120]], [-118, -81, 120]), ([[141, 87, -220, 167, 98, -180, 177, -30, 52, 203, -155, 172, 4, 92]], [141, 87, -220, 167, 98, -180, 177, -30, 52, 203, -155, 172, 4, 92]), ([[-101, -156, -120, -183, 166, -27, -14, -43, -94, -188, -170, -237]], [-101, -156, -120, -183, 166, -27, -14, -43, -94, -188, -170, -237]), ([[-203, 133, -78, 35, -253, 143, -249, -223, 203, -182, 35, -248, 186, 85, 169, -157]], [-203, 133, -78, 35, -253, 143, -249, -223, 203, -182, 35, -248, 186, 85, 169, -157])]

# Format input correctly
def decode(example):
    print("DECODE:", example)
    inputs = [Value.construct(x) for x in example[0]]
    output = Value.construct(example[1])
    return inputs, output

print("TEST1:", examples[0][0])
print("TEST2:", examples[0][1])
examples2 = [decode(x) for x in examples]

print("VALUE CONSTRUCTION EXAMPLES2", examples2)

# Depth-first search (DFS)
start = time.time()
solution, steps_used = dfs(examples2, 2, ctx, np.inf)
end = time.time()

if solution:
    solution = solution.prefix

# Print DFS results
print("DFS result:", solution)
print("Execution time:", end - start)
print("Steps used:", steps_used)

# Sort and add enumerative search
start = time.time()
solution, steps_used = sort_and_add(examples2, 2, ctx, np.inf)
end = time.time()

if solution:
    solution = solution.prefix

# Print Sort and add results
print("Sort and add result:", solution)
print("Execution time:", end - start)
print("Steps used:", steps_used)
"""
# EOF
