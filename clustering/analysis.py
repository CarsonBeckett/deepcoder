#from __future__ import print_function

"""

Filip Hörnsten

Process and analyse the output from the artefact
Calculate and output the min/max values, the mean, standard deviation, confidence interval (95%) and margin of error

"""
from itertools import islice
from fileinput import input as inp
import sys
import numpy as np, scipy.stats as st

def process_output(index, in_file):
    samples = []

    clust_exec_sample = []
    clust_wall_sample = []
    total_exec_sample = []
    total_wall_sample = []

    dfs_exec_sample = []
    dfs_wall_sample = []
    saa_exec_sample = []
    saa_wall_sample = []
    
    with open (in_file, 'r') as file:
        # DeepCoder output file has different format
        if (index == 5):
            for lineno, line in enumerate(file, start=1):
            # Print general information about DFS
                if(lineno < 9):
                    print(line, end="")
                # Analyse DFS search execution time
                elif(10 <= lineno <= 39):
                    dfs_exec_sample.append(float(line))
                # Analyse DFS search wall time
                elif(42 <= lineno <= 71):
                    dfs_wall_sample.append(float(line))
                # Print general information about sort and add
                elif(71 <= lineno <= 79):
                    print(line, end="")
                # Analyse sort and add search execution time
                elif(81 <= lineno <= 110):
                    saa_exec_sample.append(float(line))
                # Analyse sort and add search wall time
                elif(113 <= lineno <= 142):
                    saa_wall_sample.append(float(line))
                
        # Clustering scenarios
        else:
            for lineno, line in enumerate(file, start=1):
                # Print general scenario information
                if(lineno < 6):
                    print(line, end="")
                # Analyse clustering execution time
                elif(7 <= lineno <= 36):
                    clust_exec_sample.append(float(line))
                # Analyse clustering wall time
                elif(39 <= lineno <= 68):
                    clust_wall_sample.append(float(line))
                # Analyse total scenario execution time
                elif(71 <= lineno <= 100):
                    total_exec_sample.append(float(line))
                # Analyse total scenario wall time
                elif(103 <= lineno <= 132):
                    total_wall_sample.append(float(line))
                

    if(index == 5):
        samples = dfs_exec_sample, dfs_wall_sample, saa_exec_sample, saa_wall_sample
    else:
        samples = clust_exec_sample, clust_wall_sample, total_exec_sample, total_wall_sample
        
    return samples

def analyse_output(sample):
    min_value = np.amin(sample)
    print("Min:", min_value)

    max_value = np.amax(sample)
    print("Max:", max_value)

    mean = np.mean(sample, dtype=np.float64)
    print("Mean:", mean)
    
    std_deviation = np.std(sample, dtype=np.float64)
    print("Standard deviation:", std_deviation)

    print("Length:", len(sample))
    
    # Calculate the 95% confidence interval (confidence level = 0,95, alpha = 0,05)
    confidence_interval = st.t.interval(0.95, len(sample), loc=np.mean(sample), scale=st.sem(sample))
    print("Confidence interval:", confidence_interval)

    error_margin = (confidence_interval[1] - confidence_interval[0]) / 2
    print("Margin of error:", error_margin, "\n")

def perform_statistical_analysis():
    files = ['scenario1_output.data', 'scenario2_output.data', 'scenario3_output.data', 'scenario4_output.data', 'scenario5_output.data', 'ips_output.data']
    
    for file_index, file in enumerate(files):
        samples = process_output(file_index, file)
    
        for sample_index, sample in enumerate(samples):
            if(file_index == 5):
                if(sample_index == 0 or sample_index == 2):
                    print("Execution time:")
                elif(sample_index == 1 or sample_index == 3):
                    print("Wall time:")
            else:
                if(sample_index == 0):
                    print("Execution time of best clustering:")
                elif(sample_index == 1):
                    print("Wall time of best clustering:")
                elif(sample_index == 2):
                    print("Total scenario execution time:")
                elif(sample_index == 3):
                    print("Total scenario wall time:")
                
            analyse_output(sample)
    
perform_statistical_analysis()
