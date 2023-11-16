from numpy.random import seed
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import warnings
# Set the desired action for the warning
warnings.filterwarnings("ignore")

# seed the random number generator
#name = sys.argv[1] # data file
seed(1)
instancias = ["small","tsplib"]
for k in instancias:
#for i in range(7,8):
    name = f"output/gap_dl_short_{k}.txt"
    data = [[] for j in range(2)]
    f = open(name, 'r')
    for line in f:
        temp = line.split(",")
        data[0].append(float(temp[0].strip()))
        data[1].append(float(temp[1].strip()))
    f.close()

    alpha = 0.05
    stat, p = wilcoxon(data[0], data[1])#, zero_method='pratt', correction=True)
    print(f"dl_short {k}: p={p}")
print()