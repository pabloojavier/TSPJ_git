import gurobipy as gp
from gurobipy import GRB
import sys
import warnings
import networkx as nx
warnings.filterwarnings("ignore")

from Source.Problem import Problem
from Source.MathematicalModel import MathematicalModel as MILP

def CUT_fractional_callback(modelo: gp.Model, where):
    n = modelo._n
    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        return
    
    if case1:
        xval = modelo.cbGetSolution(modelo._xvars)
    elif case2:
        xval = modelo.cbGetNodeRel(modelo._xvars)
    
    edges_used = [ (i,j) for i,j in modelo._DG.edges if xval[i,j] > modelo._epsilon ]
    DG_support = modelo._DG.edge_subgraph( edges_used )

    if not nx.is_strongly_connected( DG_support ):
        for component in nx.strongly_connected_components( DG_support ):
            complement = [ i for i in modelo._DG.nodes if i not in component ]
            modelo._callback_count +=1 
            modelo.cbLazy( gp.quicksum( modelo._xvars[i,j] for i in component for j in complement ) >= 1 )

    else:
        for i,j in modelo._DG.edges:
            modelo._DG.edges[i,j]['capacity'] = xval[i,j]

        s = 0
        for t in range(1,n):
            (cut_value, node_partition) = nx.minimum_cut( modelo._DG, _s=s, _t=t )
            if cut_value < 1 - modelo._epsilon:
                S = node_partition[0]
                T = node_partition[1]
                modelo._callback_count +=1 
                modelo.cbLazy( gp.quicksum( modelo._xvars[i,j] for i in S for j in T ) >= 1 )
                return

def DFJ_fractional_callback(modelo:gp.Model, where):

    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        return
    
    if case1:
        xval = modelo.cbGetSolution(modelo._xvars)
    elif case2:
        xval = modelo.cbGetNodeRel(modelo._xvars)
    
    edges_used = [ (i,j) for i,j in modelo._DG.edges if xval[i,j] > modelo._epsilon ]
    DG_support = modelo._DG.edge_subgraph( edges_used )
    if not nx.is_strongly_connected( DG_support ):
        for component in nx.strongly_connected_components( DG_support ):
            modelo._callback_count +=1 
            modelo.cbLazy( gp.quicksum( modelo._xvars[i,j] for i in component for j in component if i!=j ) <= len(component) - 1 )
    else:
        for i,j in modelo._DG.edges:
            modelo._DG.edges[i,j]['capacity'] = xval[i,j]

        s = 0
        for t in range(1,modelo._n):
            (cut_value, node_partition) = nx.minimum_cut( modelo._DG, _s=s, _t=t )
            if cut_value < 1 - modelo._epsilon:
                S = node_partition[0]
                modelo._callback_count +=1 
                modelo.cbLazy( gp.quicksum( modelo._xvars[i,j] for i in S for j in S if i!=j ) <= len(S) - 1 )
                return

def subtourelim(modelo:gp.Model, donde):
    n = modelo._n

    if donde == GRB.Callback.MIPNODE  and ( modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL ):
        valoresX = modelo.cbGetNodeRel(modelo._xvars)
        tour = [i for i in range(n+1)]
        subtour_method(tour, valoresX,n)
        if len(tour) < n:
            modelo._callback_count +=1 
            tour2 = [i for i in range(n) if i not in tour]
            modelo.cbLazy(gp.quicksum(modelo._xvars[i, j] for i in tour for j in tour2) >= 1)

def subtour_method(subruta, vals,n):
    # obtener una lista con los arcos parte de la solucións
    arcos = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    noVisitados = list(range(n))
    while noVisitados: # true if list is non-empty
        ciclo = []
        vecinos = noVisitados
        while vecinos:
            actual = vecinos[0]
            ciclo.append(actual)
            noVisitados.remove(actual)
            vecinos = [j for i, j in arcos.select(actual, '*') if j in noVisitados]
        if len(subruta) > len(ciclo):
            subruta[:] = ciclo
     
size = "tsplib"
instance = 1
subtour = "WC"
initial_solution = False
output = False
callback = None
bounds = False

instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
configs = ("BASE","GG","GG+bounds","mycallback","dfj","cut")
for i,cb in enumerate((1,2,3,subtourelim,DFJ_fractional_callback,CUT_fractional_callback)):
    print(configs[i]+":")
    for instance in instancias:
        MM = MILP(size,instance)
        MM.callback = cb
        if cb == 1:
            MM.callback = None
            MM.subtour = "WC"
            MM.bounds = False
        elif cb == 2:
            MM.callback = None
            MM.subtour = "GG"
            MM.bounds = False
        elif cb == 3:
            MM.callback = None
            MM.subtour = "GG"
            MM.bounds = True
        else:
            MM.callback = cb
            MM.subtour = "WC"
            MM.bounds = False

        MM.output = output
        MM.initial_solution = initial_solution
        MM.run()
        MM.print_results()
    print("")    




