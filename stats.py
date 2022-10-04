import sys
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare, wilcoxon, shapiro, kruskal, rankdata
from statistics import mean
import itertools
import pandas as pd
import numpy as np

# seed the random number generator
name = sys.argv[1] # data file

seed(1)

data = [[] for i in range(5)]
#read output
f = open(name, 'r')
for line in f:
        temp = line.split(" ")
        data[0].append(float(temp[0].strip()))
        data[1].append(float(temp[1].strip()))
        data[2].append(float(temp[2].strip()))
        data[3].append(float(temp[3].strip()))
        data[4].append(float(temp[4].strip()))
f.close()

# print(data[0])
# print(data[1])
# print(data3)
# print(data4)
# print(shapiro(data[0]))
# f_test = friedmanchisquare(data[0], data[1], data[2], data[3])
# print(f_test)
# wilc_test = [wilcoxon(i, j) for i,j in itertools.combinations([0, 1, 2, 3], 2)]
# print(wilc_test)
# w_res = pd.DataFrame(wilc_test)
# w_res['test'] = ["wilcoxon " + i+" vs "+j for i,j in itertools.combinations(data)]

#check this page: https://www.schramm.cc/skripten/stat/statistics-calculator.php
for i in range(0, 5):
        print(mean(data[i]), end=" ")
print()
alpha = 0.05
for i in range(0, 5):
        if i == 0:
            print("-", end=" ")
        else:
            #print(i, mean(data[i]))
            stat, p = wilcoxon(data[0], data[i])#, zero_method='pratt', correction=True)
            #print('Wilcoxon: Statistics=%.20f, p=%.20f' % (stat, p))
            # interpret
            print(p, end=" ")

            # if p > alpha:
	    #         print('\tSame distribution (fail to reject H0)')
            # else:
	    #         print('\tDifferent distribution (reject H0)')
            # stat, p = kruskal(data[0], data[i])
            # print('Kruskal: Statistics=%.3f, p=%.3f' % (stat, p))
            # # interpret
            # if p > alpha:
	    #         print('\tSame distributions (fail to reject H0)')
            # else:
	    #         print('\tDifferent distributions (reject H0)')
print()
# import scikit_posthocs as sp

# # H, p = friedmanchisquare(*data)
# # print(H, p)
# # ranks = np.array([rankdata(-p) for p in data])
# import scipy.stats as ss
# import statsmodels.api as sa
# import scikit_posthocs as sp
# df = sa.datasets.get_rdataset('iris').data
# print(df)
# print(sp.posthoc_conover(df, val_col='Sepal.Width', group_col='Species', p_adjust = 'holm'))

# performances = pd.read_csv('data.csv')
# print(performances)
# algorithms_names = performances.drop('dataset', axis=1).columns
# performances_array = performances[algorithms_names].values
# H, p = friedmanchisquare(*performances_array)
# print(H, p)
# ranks = np.array([rankdata(-p) for p in performances_array])
# average_ranks = np.mean(ranks, axis=0)
# print(sp.posthoc_ttest(data, p_adjust = 'holm'))
# a = [data[0], data[3]]
# print(sp.posthoc_dunn(a, p_adjust = 'holm'))
# import numpy as np
# import matplotlib.pyplot as plt
# plt.boxplot((data[0], data[1],  data[2],  data[3],  data[4]), showfliers=False)
# plt.show()
