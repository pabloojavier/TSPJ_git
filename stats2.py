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
#instancias = ["large","medium","small","tsplib"]
for i in range(1,7):
    name = f"/Users/pablogutierrezaguirre/Desktop/TSPJ_git/output/wilcoxon_exp/gap_wilcoxon_exp{i}.txt"
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