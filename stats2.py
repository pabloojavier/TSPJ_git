import sys
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare, wilcoxon, shapiro, kruskal, rankdata
from statistics import mean
import itertools
import pandas as pd
import numpy as np

# seed the random number generator
#name = sys.argv[1] # data file
seed(1)
instancias = ["large","medium","small","tsplib"]
for i in range(2,3):
    for k in instancias:
        name = f"output/wilcoxon_bls/gap_wilcoxon_bls{i}_{k}.txt"
        data = [[] for j in range(2)]
        f = open(name, 'r')
        for line in f:
            temp = line.split(" ")
            data[0].append(float(temp[0].strip()))
            data[1].append(float(temp[1].strip()))
        f.close()

        alpha = 0.05
        stat, p = wilcoxon(data[0], data[1])#, zero_method='pratt', correction=True)
        print(p)
print()