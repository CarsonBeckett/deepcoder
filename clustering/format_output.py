#from __future__ import print_function

"""

Filip Hörnsten

http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

"""

from itertools import islice
from fileinput import input as inp
import sys

with open ('results.data', 'r') as in_file:
    clust_exec = None
    clust_wall = None
    total_exec = None
    total_wall = None
    ips_time = None
    ips_wall = None
    for line in in_file:
        #print("DEBUG:", line)
        if line.rstrip() == "Sort and add result: LIST|MAXIMUM,0":
            #print("HELLO")
            #in_file.seek(0) # reset pointer
            #print("Execution time of best clustering:")
            clust_exec = islice(in_file, 0, 1) # get lines 3-5, o based indexing
            clust_wall = islice(in_file, 1, 2)
            total_exec = islice(in_file, 3, 4)
            total_wall = islice(in_file, 4, 5)

            # Clustering results
            
            #for item in clust_exec:
            #    print(item, end="")

            #for item in clust_wall:
            #    print(item, end="")
                
            #for item in total_exec:
            #    print(item, end="")

            #for item in total_wall:
            #    print(item, end="")

            # IPS results

            ips_time = islice(in_file, 0, 1) # get lines 3-5, o based indexing
            ips_wall = islice(in_file, 1, 2)

            #for item in ips_time:
            #    print(item, end="")

            for item in ips_wall:
                print(item, end="")
            
            #break
            #print()
