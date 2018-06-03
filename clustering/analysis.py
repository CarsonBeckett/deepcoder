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
import matplotlib.pyplot as plt

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

def plot_analysis(plot_information):
    # Plot analysis of clustering
    clusteringMeansList =[]
    clusteringErrorList = []
    for i in range(0, len(plot_information)-7, 4):
        clusteringMeansList.append(plot_information[i][0])
        clusteringErrorList.append(plot_information[i][1])

    clusteringMeans = tuple(clusteringMeansList)
    clusteringError = tuple(clusteringErrorList)
    print("Clustering means:", clusteringMeans)
    print("Clustering errors:", clusteringError)

    scenarioMeansList =[]
    scenarioErrorList = []
    for i in range(2, len(plot_information)-5, 4):
        scenarioMeansList.append(plot_information[i][0])
        scenarioErrorList.append(plot_information[i][1])

    scenarioMeans = tuple(scenarioMeansList)
    scenarioError = tuple(scenarioErrorList)
    print("Scenario means:", scenarioMeans)
    print("Scenario errors:", scenarioError)

    scenarioMeans = scenarioMeans[:5]
    print("New scenario means:", scenarioMeans)
    scenarioError = scenarioError[:5]
    print("New scenario error:", scenarioError)
    
    N_CLUSTERING = 1
    ind = np.arange(N_CLUSTERING)    # the x locations for the groups
    
    width = 0.35       # the width of the bars: can also be len(x) sequence

    plot_format = [(0, 0.225, 0.01), (0, 0.235, 0.01), (0, 0.245, 0.02), (0, 0.365, 0.02), (0, 3.45, 0.2)]

    for i in range(5):
        p1 = plt.bar(ind, clusteringMeans[i], width, yerr=clusteringError[i])
        #p2 = plt.bar(ind, scenarioMeans, width, bottom=clusteringMeans, yerr=scenarioError)
        p2 = plt.bar(ind, scenarioMeans[i], width, bottom=clusteringMeans[i], yerr=scenarioError[i])

        start, stop, tick = plot_format[i]
        display = ('Scenario ' + str(i+1),)
        plt.ylabel('Run time in seconds')
        plt.title('Average run times with confidence intervals')
        plt.xticks(ind, display)
        plt.yticks(np.arange(start, stop, tick))
        #plt.legend([])
        plt.legend((p1[0], p2[0]), ('Clustering run time', 'Total scenario run time'))

        plt.show()

    # Plot analysis of IPS
    dfs_mean, dfs_error = plot_information[20]
    saa_mean, saa_error = plot_information[22]
    print("DFS mean:", dfs_mean)
    print("DFS error:", dfs_error)
    print("Sort and add mean:", saa_mean)
    print("Sort and add error:", saa_error)

    N_IPS = 2
    ind_ips = np.arange(N_IPS)

    means = ()
    error = ()

    means += (dfs_mean,) + (saa_mean,)
    error += (dfs_error,) + (saa_error,)
    print("Means:", means)
    print("Error:", error)
    
    p1 = plt.bar(ind_ips, means, width, yerr=error)
    #p2 = plt.bar(ind, scenarioMeans, width, bottom=clusteringMeans, yerr=scenarioError)
    #p2 = plt.bar(ind_ips, saa_mean, width, yerr=saa_error)

    plt.ylabel('Run time in seconds')
    plt.title('Average run times with confidence intervals')
    plt.xticks(ind_ips, ('DFS', 'Sort and add'))
    plt.yticks(np.arange(0, 0.0015, 0.0001))
    #plt.legend([])
    #plt.legend((p1[0], p2[0]), ('Total scenario run time', 'Clustering run time'))

    plt.show()
    
    """plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    y_pos = np.arange(len(people))
    performance = 3 + 10 * np.random.rand(len(people))
    error = np.random.rand(len(people))

    ax.barh(y_pos, performance, xerr=error, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Performance')
    ax.set_title('How fast do you want to go today?')

    plt.show()"""

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

    return mean, error_margin

def perform_statistical_analysis():
    files = ['scenario1_output.data', 'scenario2_output.data', 'scenario3_output.data', 'scenario4_output.data', 'scenario5_output.data', 'ips_output.data']

    plot_information = []
    
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
                
            plot_information.append(analyse_output(sample))

    print(plot_information)            
    plot_analysis(plot_information)
    
perform_statistical_analysis()
#plot_analysis()
