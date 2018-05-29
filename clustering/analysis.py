#from __future__ import print_function

"""

Filip Hörnsten

http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

"""
from itertools import islice
from fileinput import input as inp
import sys
import numpy as np, scipy.stats as st

def perform_statistical_analysis():
    #sample = np.empty([1,1])
    files = ['scenario1_output.data', 'scenario2_output.data', 'scenario3_output.data', 'scenario4_output.data', 'scenario5_output.data']
    for file in files:
        samples = []

        clust_exec_sample = []
        clust_wall_sample = []
        total_exec_sample = []
        total_wall_sample = []
        
        with open (file, 'r') as file:
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
                elif(71 <= lineno <= 100):
                    total_exec_sample.append(float(line))
                elif(103 <= lineno <= 132):
                    total_wall_sample.append(float(line))
                    

        samples = clust_exec_sample, clust_wall_sample, total_exec_sample, total_wall_sample
        #print(clust_exec_sample)
        
        for sample in samples:
            if(sample == clust_exec_sample):
                print("Execution time of best clustering:")
            elif(sample == clust_wall_sample):
                print("Wall time of best clustering:")
            elif(sample == total_exec_sample):
                print("Total scenario execution time:")
            elif(sample == total_wall_sample):
                print("Total scenario wall time:")

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
            #np.set_printoptions(suppress=True,
            #formatter={'float_kind':'{:2.25f}'.format})
            print("Margin of error:", error_margin, "\n")
    
perform_statistical_analysis()
