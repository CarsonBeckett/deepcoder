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
import unittest
from deepcoder import context
from deepcoder.search import dfs, sort_and_add
from deepcoder.dsl import impl
from deepcoder.dsl import types
from deepcoder.dsl.function import OutputOutOfRangeError
from deepcoder.dsl.program import Program, prune, get_unused_indices
from deepcoder.dsl.value import Value, IntValue, ListValue

def decode_example(example):
        #print("DECODE:", example)
        inputs = [Value.construct(x) for x in example[0]]
        output = Value.construct(example[1])
        return inputs, output

# Test section
def test_leader_election(solution, examples):
        prefix = str(solution)
        program = Program.parse(prefix)

        consistent = True
        for tuples in examples:
            #print("tuples:", tuples)
            inputs, output = tuples
            #print("Inputs:", inputs)
            #print("Output:", output)
            #print("stripped:", inputs[0])
            #print("TYPE:", type(inputs[0]))
            #raw_input = []
            #for i in range(len(inputs[0])):
            #    raw_input.append(inputs[0][i])
            #raw_input = inputs[0]
            #raw_input2 = raw_input.val
            #print("Raw input:", raw_input2)
            
            expected = IntValue(monarchical_leader_election(inputs[0].val))
            actual = program(inputs[0])
            #assertEqual(dfs_actual, dfs_expected)
            #print("DFS expected:", dfs_expected)
            #print("DFS actual:", dfs_actual)
            if (actual != expected):
                consistent = False
                print("DeepCoder program inconsistent with ground truth program")
                break
            #assertEqual(dfs_program.toprefix(), prefix)

        return consistent

def main():
    # Run the clustering algorithm
    scenarios = cluster_nodes(visualisation=True)
    #print("MAIN SCENARIOS:", scenarios)
    #print("TEST1", scenarios[0][0])
    # Extract the clusters in each scenario from a list of all scenarios
    scenarioRanks = []
    ranks = []
    clusterList = []
    for scenario in scenarios:
        for clusters in scenario:
            #print("CLUSTERSMAIN", clusters)
            scenarioRanks.append(rank(clusters))

        ranks.append(scenarioRanks)

    #print("Ranks:", ranks)
    #print("Length:", len(ranks))
    #print("Ranks for one scenario:", ranks[0])

    # Package the ranks of one cluster together with its leader to prepare for DeepCoder processing
    
    examples = []
    for scenario in ranks:
        for clusterRanks in scenario:
            # Strip away node ID / index in sample from a copy of the list
            strippedRanks = []
            #print("TEST999:", clusterRanks)
            for i in range(len(clusterRanks)):
                strippedRanks.append(clusterRanks[i][1])
            
            # Ground truth / oracle
            leader = monarchical_leader_election(strippedRanks)

            # Shuffle the order of the list of ranks to avoid DeepCoder search finding incorrect program
            # E.g. the format obtained from the function 'rank' is sorted meaning DeepCoder can incorrectly believe
            # that getting the last element (tail) of the list is also correct.
            numpy.random.shuffle(strippedRanks)
            #print("Shuffled ranks:", strippedRanks)
            
            #Build the input-output tuple
            examples.append(([strippedRanks], leader))
            #print("TEST2", strippedRanks)
            #print("IndexLeader:", len(strippedRanks)-1)
            #print("Leader:", leader)

    #print("Full examples:", examples)
    #print("TEST3:", examples[0])
    
    # Preprocessing
    decoded_examples = [decode_example(x) for x in examples]
    predictions = numpy.zeros(len(impl.FUNCTIONS))
    scores = dict(zip(impl.FUNCTIONS, predictions))
    ctx = context.Context(scores)
    
    #print("VALUE CONSTRUCTION EXAMPLES", decoded_examples)

    # Pass formatted rank and elected leader as input-output examples to DeepCoder

    # Depth-first search (DFS)
    dfs_start = time.time()
    dfs_solution, dfs_steps_used = dfs(decoded_examples, 2, ctx, numpy.inf)
    dfs_end = time.time()

    # Sort and add enumerative search
    saa_start = time.time()
    saa_solution, saa_steps_used = sort_and_add(decoded_examples, 2, ctx, numpy.inf)
    saa_end = time.time()

    # Compare the elected leader from running the program inferred by DeepCoder to the ground truth from the oracle
    if dfs_solution:
        dfs_solution = dfs_solution.prefix
        print("\nSynthesised program using DFS consistent with ground truth:", test_leader_election(dfs_solution, decoded_examples))
    else:
        print("\nNo solution found with DFS")
        
    if saa_solution:
        saa_solution = saa_solution.prefix
        print("Synthesised program using sort and add consistent with ground truth:", test_leader_election(saa_solution, decoded_examples))

    else:
        print("No solution found with sort and add")

    # Print DFS results
    print("\nDFS result:", dfs_solution)
    print("Execution time:", dfs_end - dfs_start)
    print("Steps used:", dfs_steps_used)

    # Print Sort and add results
    print("\nSort and add result:", saa_solution)
    print("Execution time:", saa_end - saa_start)
    print("Steps used:", saa_steps_used)
        
main()

# EOF
