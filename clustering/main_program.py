"""!

@brief Filip Hörnsten's Bachelor degree project 2018

@authors Filip Hörnsten
@date 2018
@copyright GNU Public License
"""

from leader_election import rank, monarchical_leader_election
from kmedoids_cluster_nodes import cluster_nodes, silhouette_value
import numpy
import time
from deepcoder import context
from deepcoder.search import dfs, sort_and_add
from deepcoder.dsl import impl
from deepcoder.dsl import types
from deepcoder.dsl.value import Value

def main():
    # Run the clustering algorithm
    scenarios = cluster_nodes()
    #print("MAIN SCENARIOS:", scenarios)
    print("TEST1", scenarios[0][0])
    # Extract the clusters in each scenario from a list of all scenarios
    scenarioRanks = []
    ranks = []
    clusterList = []
    for scenario in scenarios:
        for clusters in scenario:
            print("CLUSTERSMAIN", clusters)
            scenarioRanks.append(rank(clusters))

        ranks.append(scenarioRanks)

    print("Ranks:", ranks)
    print("Length:", len(ranks))
    print("Ranks for one scenario:", ranks[0])

    # Package the ranks of one cluster together with its leader to prepare for DeepCoder processing
    
    examples = []
    for scenario in ranks:
        for clusterRanks in scenario:
            # Strip away node ID / index in sample from a copy of the list
            strippedRanks = []
            for i in range(len(clusterRanks)):
                strippedRanks.append(clusterRanks[i][1])
            
            # Ground truth / oracle
            leader = monarchical_leader_election(strippedRanks)

            # Shuffle the order of the list of ranks to avoid DeepCoder search finding incorrect program
            # E.g. the format obtained from the function 'rank' is sorted meaning DeepCoder can incorrectly believe
            # that getting the last element (tail) of the list is also correct.
            numpy.random.shuffle(strippedRanks)
            print("Shuffled ranks:", strippedRanks)
            
            #Build the input-output tuple
            examples.append((strippedRanks, leader))
            print("TEST2", strippedRanks)
            print("IndexLeader:", len(strippedRanks)-1)
            print("Leader:", leader)

    print("Full examples:", examples)
    print("TEST3:", examples[0])
    
    # Preprocessing
    predictions = numpy.zeros(len(impl.FUNCTIONS))
    scores = dict(zip(impl.FUNCTIONS, predictions))
    ctx = context.Context(scores)

    def decode(example):
        print("DECODE:", example)
        inputs = [Value.construct(x) for x in example[0]]
        output = Value.construct(example[1])
        return inputs, output

    examples2 = [decode(x) for x in examples]
    print("VALUE CONSTRUCTION EXAMPLES2", examples2)

    # Pass formatted rank and elected leader as input-output examples to DeepCoder

    # Depth-first search (DFS)
    start = time.time()
    solution, steps_used = dfs(examples2, 2, ctx, numpy.inf)
    end = time.time()

    if solution:
        solution = solution.prefix

    # Print DFS results
    print("DFS result:", solution)
    print("Execution time:", end - start)
    print("Steps used:", steps_used)

    # Sort and add enumerative search
    start = time.time()
    solution, steps_used = sort_and_add(examples2, 2, ctx, numpy.inf)
    end = time.time()

    if solution:
        solution = solution.prefix

    # Print DFS results
    print("Sort and add result:", solution)
    print("Execution time:", end - start)
    print("Steps used:", steps_used)
    
main()

# EOF
