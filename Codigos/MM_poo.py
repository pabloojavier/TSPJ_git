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

size = "medium"
instance = "1"
output = True
subtour = "dl"
initial_sol = True
callback = "subtourelim1"
bounds = True
new_formulation = True
time_limit = 1800

for i in range(len(opts)):
    if opts[i][0][1:] == "size":  size  = opts[i][1]
    elif opts[i][0][1:] == "instance": 
        try:
            instance = int(opts[i][1])  
        except ValueError:
            instance = opts[i][1]  
    elif   opts[i][0][1:] == "initial_sol": initial_sol = False if opts[i][1].lower() == "false" else True
    elif   opts[i][0][1:] == "output": output =  False if opts[i][1].lower() == "false" else True
    elif   opts[i][0][1:] == "bounds": bounds = False if opts[i][1].lower() == "false" else True
    elif   opts[i][0][1:] == "subtour": subtour = opts[i][1]
    elif   opts[i][0][1:] == "callback": callback = opts[i][1]
    elif   opts[i][0][1:] == "newformulation": new_formulation =  False if opts[i][1].lower() == "false" else True

MM = MILP(size,
          instance,
          output,
          subtour,
          initial_sol,
          callback,
          bounds,
          new_formulation,
          time_limit
          )
MM.run()
MM.print_results()
