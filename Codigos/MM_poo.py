import gurobipy as gp
from gurobipy import GRB
import sys
import warnings
import networkx as nx
import time
warnings.filterwarnings("ignore")

from Source.Problem import Problem
from Source.MathematicalModel import MathematicalModel as MILP

def CUT_integer_separation(m:gp.Model, where):
    initial = time.time()
    # check if LP relaxation at this branch-and-bound node has an integer solution
    if where == GRB.Callback.MIPSOL: 
        
        # retrieve the LP solution
        xval = m.cbGetSolution(m._xvars)
        
        # which edges are selected?
        edges_used = [ (i,j) for i,j in m._DG.edges if xval[i,j] > 0.5 ]
        
        # create support graph
        DG_soln = m._DG.edge_subgraph( edges_used )
        
        # if solution is not connected, add a (violated) CUT constraint for each subtour
        if not nx.is_strongly_connected( DG_soln ):
            for component in nx.strongly_connected_components( DG_soln ):
                m._callback_count +=1 
                complement = [ i for i in DG_soln.nodes if i not in component ]
                m.cbLazy( gp.quicksum( m._x[i,j] for i in component for j in complement ) >= 1 )
    m._callback_time += time.time()-initial

def CUT_naive_fractional_separation(m:gp.Model, where):
    initial = time.time()
    # We must *separately* handle these two cases of interest:
    #    1. We encounter an integer point that might replace our incumbent
    #    2. We encounter a fractional point *that is LP optimal*
    #
    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        return
    
    # retrieve the LP solution
    if case1:
        xval = m.cbGetSolution(m._xvars)
    elif case2:
        xval = m.cbGetNodeRel(m._xvars)
        
    # which edges are selected (in whole or in part)?
    DG = m._DG
    edges_used = [ (i,j) for i,j in DG.edges if xval[i,j] > m._epsilon ]

    # check if any CUT inequalities are violated (by solving some min cut problems)
    for i,j in DG.edges:
        DG.edges[i,j]['capacity'] = xval[i,j]

    s = 0
    for t in range(1,m._n):
        (cut_value, node_partition) = nx.minimum_cut( DG, _s=s, _t=t )
        # print("cut_value =",cut_value)
        if cut_value < 1 - m._epsilon:
            m._callback_count +=1 
            S = node_partition[0]  # 'left' side of the cut
            T = node_partition[1]  # 'right' side of the cut
            m.cbLazy( gp.quicksum( m._xvars[i,j] for i in S for j in T ) >= 1 )
            m._callback_time += time.time()-initial
            return

def CUT_smarter_fractional_separation(m:gp.Model, where):
    initial = time.time()
    # We must *separately* handle these two cases of interest:
    #    1. We encounter an integer point that might replace our incumbent
    #    2. We encounter a fractional point *that is LP optimal*
    #
    n = m._n
    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        m._callback_time += time.time()-initial
        return
    
    # retrieve the LP solution
    if case1:
        xval = m.cbGetSolution(m._xvars)
    elif case2:
        xval = m.cbGetNodeRel(m._xvars)
    
    DG = m._DG
    # if the support graph is disconnected, then finding violated cuts is easy!
    edges_used = [ (i,j) for i,j in DG.edges if xval[i,j] > m._epsilon ]
    DG_support = DG.edge_subgraph( edges_used )
    if not nx.is_strongly_connected( DG_support ):
        for component in nx.strongly_connected_components( DG_support ):
            m._callback_count +=1 
            complement = [ i for i in DG.nodes if i not in component ]
            m.cbLazy( gp.quicksum( m._xvars[i,j] for i in component for j in complement ) >= 1 )
    else:
        # check if any CUT inequalities are violated (by solving some min cut problems)
        for i,j in DG.edges:
            DG.edges[i,j]['capacity'] = xval[i,j]

        s = 0
        for t in range(1,n):
            (cut_value, node_partition) = nx.minimum_cut( DG, _s=s, _t=t )
            # print("cut_value =",cut_value)
            if cut_value < 1 - m._epsilon:
                m._callback_count +=1 
                S = node_partition[0]  # 'left' side of the cut
                T = node_partition[1]  # 'right' side of the cut
                m.cbLazy( gp.quicksum( m._xvars[i,j] for i in S for j in T ) >= 1 )
                m._callback_time += time.time()-initial
                return

def DFJ_integer_separation(m:gp.Model, where):
    initial = time.time()
    # check if LP relaxation at this branch-and-bound node has an integer solution
    if where == GRB.Callback.MIPSOL: 
        
        # retrieve the LP solution
        xval = m.cbGetSolution(m._xvars)
        
        # which edges are selected?

        edges_used = [ (i,j) for i,j in m._DG.edges if xval[i,j] > 0.5 ]
        
        # create support graph
        DG_soln = m._DG.edge_subgraph( edges_used )
        
        # if solution is not connected, add a (violated) DFJ constraint for each subtour
        if not nx.is_strongly_connected( DG_soln ):
            for component in nx.strongly_connected_components( DG_soln ):
                m._callback_count +=1 
                m.cbLazy( gp.quicksum( m._xvars[i,j] for i in component for j in component if i!=j ) <= len(component) - 1 )
    m._callback_time += time.time()-initial

def DFJ_naive_fractional_separation(m:gp.Model, where):
    initial = time.time()
    # We must *separately* handle these two cases of interest:
    #    1. We encounter an integer point that might replace our incumbent
    #    2. We encounter a fractional point *that is LP optimal*
    #
    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        m._callback_time += time.time()-initial
        return
    
    # retrieve the LP solution
    if case1:
        xval = m.cbGetSolution(m._xvars)
    elif case2:
        xval = m.cbGetNodeRel(m._xvars)
        
    # which edges are selected (in whole or in part)?
    DG = m._DG
    edges_used = [ (i,j) for i,j in DG.edges if xval[i,j] > m._epsilon ]

    # check if any CUT inequalities are violated (by solving some min cut problems)
    for i,j in DG.edges:
        DG.edges[i,j]['capacity'] = xval[i,j]

    s = 0
    for t in range(1,m._n):
        (cut_value, node_partition) = nx.minimum_cut( DG, _s=s, _t=t )
        # print("cut_value =",cut_value)
        if cut_value < 1 - m._epsilon:
            m._callback_count +=1 
            S = node_partition[0]  # 'left' side of the cut
            m.cbLazy( gp.quicksum( m._xvars[i,j] for i in S for j in S if i!=j ) <= len(S) - 1 )
    m._callback_time += time.time()-initial

def DFJ_smarter_fractional_separation(m:gp.Model, where):
    initial = time.time()
    # We must *separately* handle these two cases of interest:
    #    1. We encounter an integer point that might replace our incumbent
    #    2. We encounter a fractional point *that is LP optimal*
    #
    case1 = where == GRB.Callback.MIPSOL
    case2 = ( where == GRB.Callback.MIPNODE ) and ( m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL )
    
    if not case1 and not case2:
        m._callback_time += time.time()-initial
        return
    
    # retrieve the LP solution
    if case1:
        xval = m.cbGetSolution(m._xvars)
    elif case2:
        xval = m.cbGetNodeRel(m._xvars)
    
    # if the support graph is disconnected, then finding violated cuts is easy!
    DG = m._DG
    edges_used = [ (i,j) for i,j in DG.edges if xval[i,j] > m._epsilon ]
    DG_support = DG.edge_subgraph( edges_used )
    if not nx.is_strongly_connected( DG_support ):
        for component in nx.strongly_connected_components( DG_support ):
            m._callback_count +=1 
            m.cbLazy( gp.quicksum( m._xvars[i,j] for i in component for j in component if i!=j ) <= len(component) - 1 )
    else:
        # check if any CUT inequalities are violated (by solving some min cut problems)
        for i,j in DG.edges:
            DG.edges[i,j]['capacity'] = xval[i,j]

        s = 0
        for t in range(1,m._n):
            (cut_value, node_partition) = nx.minimum_cut( DG, _s=s, _t=t )
            # print("cut_value =",cut_value)
            if cut_value < 1 - m._epsilon:
                S = node_partition[0]  # 'left' side of the cut
                m._callback_count +=1 
                m.cbLazy( gp.quicksum( m._xvars[i,j] for i in S for j in S if i!=j ) <= len(S) - 1 )
                m._callback_time += time.time()-initial
                return

def subtourelim(modelo:gp.Model, donde):
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
    initial = time.time()
    n = modelo._n

    if donde == GRB.Callback.MIPNODE  and ( modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL ):
        valoresX = modelo.cbGetNodeRel(modelo._xvars)
        tour = [i for i in range(n+1)]
        subtour_method(tour, valoresX,n)
        if len(tour) < n:
            modelo._callback_count +=1 
            tour2 = [i for i in range(n) if i not in tour]
            modelo.cbLazy(gp.quicksum(modelo._xvars[i, j] for i in tour for j in tour2) >= 1)
    modelo._callback_time += time.time()-initial
     
size = "tsplib"
instance = "gr17"
subtour = "WC"
initial_solution = False
output = False
callback = None
bounds = False
test = [1,2,
        subtourelim,
        CUT_integer_separation,CUT_naive_fractional_separation,CUT_smarter_fractional_separation,
        DFJ_integer_separation,DFJ_naive_fractional_separation,DFJ_smarter_fractional_separation]

instancias = [["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"],
              [i for i in range(1,101)],
              [i for i in range(1,101)]]

configs = ("BASE","GG","mycallback","CUT integer","CUT naive","CUT smarter","DJF integer","DJF naive","DJF smarter")

#           #callback,bounds,subtour
options = [(subtourelim,True,"WC"),
           (None,True,"GG"),
           (CUT_naive_fractional_separation,True,"WC"),
           (CUT_smarter_fractional_separation,True,"WC")]

for cb,bool_bounds,subtour_type in options:
    print(f"{cb.__name__} and {'bounds' if bool_bounds else 'no bounds'}:")
    for j,size in enumerate(("tsplib","small","medium")):
        for instance in instancias[j]:
            MM = MILP(size,instance)
            MM.callback = cb
            MM.bounds = bool_bounds
            MM.callback = cb
            MM.subtour = subtour_type
            MM.output = output
            MM.initial_solution = initial_solution
            MM.run()
            MM.print_results()
print("")    