import gurobipy as gp
from gurobipy import GRB
import sys
import warnings
import networkx as nx
import time
warnings.filterwarnings("ignore")

from Source.MathematicalModel import MathematicalModel as MILP

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

size = "tsplib"
instance = "gr17"
callback = "None"
bounds = True
subtour = "GG"
output = False
initial_sol = True

for i in range(len(opts)):
    if opts[i][0][1:] == "size":  size  = opts[i][1]
    elif opts[i][0][1:] == "instance": 
        try:
            instance = int(opts[i][1])  
        except ValueError:
            instance = opts[i][1]  
    elif   opts[i][0][1:] == "initial_sol": initial_sol = False if opts[i][1] == "False" else True
    elif   opts[i][0][1:] == "output": output =  False if opts[i][1] == "False" else True
    elif   opts[i][0][1:] == "bounds": bounds = False if opts[i][1] == "False" else True
    elif   opts[i][0][1:] == "subtour": subtour = opts[i][1]
    elif   opts[i][0][1:] == "callback": callback = opts[i][1]


MM = MILP(size,instance)
MM.callback = callback
MM.bounds = bounds
MM.subtour = subtour
MM.output = output
MM.initial_solution = initial_sol
MM.time_limit = 1
MM.run()
MM.print_results()
