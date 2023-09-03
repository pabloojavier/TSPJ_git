import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import warnings
import networkx as nx
from typing import Dict, List, Callable, Any
from Source.Problem import Problem 
#from Problem import Problem 
import time
warnings.filterwarnings("ignore")

class MathematicalModel(Problem):
    def __init__(self,size:str,
                 instance,
                 output = False,
                 subtour : str = "GG",
                 initial_solution : bool = True,
                 callback : Callable = "None",
                 bounds: bool = True
                 ):
        """
        Initialize mathematical model with some default values

        Default values:

            output = True\n
            subtour = GG\n
            initial_solution = True\n
        """
        super().__init__(size,instance)
        self.size = size
        self.output = output
        self.subtour = subtour
        self.initial_solution = initial_solution
        self.callback = callback
        self.bounds = bounds
        self.time_limit = 1800
        self.callback_dict = {"CUT_integer_separation":MathematicalModel.CUT_integer_separation,
                              "CUT_naive_fractional_separation":MathematicalModel.CUT_naive_fractional_separation,
                              "CUT_smarter_fractional_separation":MathematicalModel.CUT_smarter_fractional_separation,
                              "DFJ_integer_separation":MathematicalModel.DFJ_integer_separation,
                              "DFJ_naive_fractional_separation":MathematicalModel.DFJ_naive_fractional_separation,
                              "DFJ_smarter_fractional_separation":MathematicalModel.DFJ_smarter_fractional_separation,
                              "subtourelim":MathematicalModel.subtourelim}

        self.compute_M()
        self.jobs = self.cities.copy()
        self.jobs_arch = [(i,k) for i in self.cities for k in self.cities]
        
        #jt_aux = np.array(list(self.JT.values()))
        self.jt_min = self.JT.replace(0,9999999).to_numpy().min()
        #self.jt_min = jt_aux[(jt_aux >= 1)].min()

    def compute_M(self):
        """
        This method computes the M value, it can be overwritting.
        """
        self.M = 0
        for i in range(self.n):
            max_t = 0
            max_tt = 0
            for j in range(self.n):
                if i!=j and self.TT[i][j]>max_t:
                    max_t = self.TT[i][j]
                
                if j!= 0 and self.JT[j][i]>max_tt:
                    max_tt = self.JT[j][i]
            self.M += max_t #+max_tt

    def create_base_model(self):
        """
        Create the initial MILP, whithout any additional constraint.
        """
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if self.output else 0)
        env.start()
        self.modelo = gp.Model(env=env)

        self.Cmax = self.modelo.addVar(vtype=GRB.CONTINUOUS,name="Cmax")

        self.x = self.modelo.addVars(self.arch, vtype=GRB.BINARY, name='x')
        self.z = self.modelo.addVars(self.jobs_arch, vtype=GRB.BINARY, name='z')

        self.TS = self.modelo.addVars(self.cities,vtype=GRB.CONTINUOUS,name="TS")

        self.modelo.setObjective(self.Cmax, GRB.MINIMIZE)

        #Restricción adicional de reforzamiento
        self.modelo.addConstr(self.Cmax >= self.jt_min + gp.quicksum(self.x[(i,j)]*self.TT[i][j] for i in self.cities for j in self.cities[1:] if i!=j))

        self.modelo.addConstrs(self.x.sum(i,'*') == 1 for i in self.cities) # Outgoing
        self.modelo.addConstrs(self.x.sum('*', j) == 1 for j in self.cities) # Incoming 

        #Restricciones
        for i in self.cities[1:self.n]:
            self.modelo.addConstr(self.Cmax >= self.TS[i] + gp.quicksum(self.z[(i,k)]*self.JT[k][i] for k in self.jobs if k!=0 ))

        for i in self.cities[1:self.n]:
            self.modelo.addConstr(self.Cmax >= self.TS[i] + self.x[(i,0)]*self.TT[0][i])

        for k in self.jobs[1:self.n]:
            self.modelo.addConstr(gp.quicksum(self.z[(i,k)] for i in self.cities if i != 0) == 1)

        for i in self.cities[1:self.n]: 
            self.modelo.addConstr(gp.quicksum(self.z[(i,k)] for k in self.jobs if k != 0) == 1)
        

        for i in self.cities: #16
            for j in self.cities[1:self.n]:
                if i!=j:
                    self.modelo.addConstr(self.TS[i] + self.TT[j][i] - (1-self.x[(i,j)])*self.M <= self.TS[j])

        
        self.modelo.Params.Threads = 1
        self.modelo.Params.TimeLimit = self.time_limit
        self.modelo._callback_time = 0
        self.modelo.update()

    def add_subtour_constraint(self):
        """
        Adds subtour constraints, it can be GG, MTZ or DL constraint
        """
        if self.subtour == "GG":
            self.y = self.modelo.addVars(self.arch,name = "y") 
            for i in self.cities[1:self.n]: #14
                self.modelo.addConstr(gp.quicksum(self.y[(i,j)] for j in self.cities if i !=j) - gp.quicksum(self.y[(j,i)] for j in self.cities if i !=j) == 1)

            for i in self.cities[1:self.n]: #15
                for j in self.cities:
                    if i!=j:
                        self.modelo.addConstr(self.y[(i,j)]<= self.n*self.x[(i,j)])
            
        elif self.subtour in ("MTZ","DL"):
            self.u = self.modelo.addVars(self.cities , vtype = GRB.CONTINUOUS , name = "u")
            if self.subtour == "MTZ":    
                for i,j in self.arch:
                    if i>0:
                        self.modelo.addConstr(self.u[i] - self.u[j] + 1 <= self.M * (1 - self.x[(i,j)]) , "MTZ(%s,%s)" %(i, j))
            
            elif self.subtour == "DL":
                for i in range(1,self.n):
                    self.modelo.addConstr(self.u[i] >= 1 + (self.n-3)*self.x[(i,0)] + gp.quicksum(self.x[(j,i)] for j in self.cities[1:] if j != i))

                    for i in range(1,self.n):
                        self.modelo.addConstr(self.u[i] <= self.n-1 - (self.n-3)*self.x[(0,i)]- gp.quicksum(self.x[(i,j)] for j in self.cities[1:] if j != i))
        
        self.modelo.update()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        
    @staticmethod
    def __sort_arch(route):
        """
        Aux method, from a list solution to a arch solution.
        """
        archs = []
        for i in range(len(route)-1):
            archs.append((str(route[i]),str(route[i+1])))
        archs.append((str(route[-1]),str(route[0])))
        return archs

    @staticmethod
    def NNJA(route,JT):
        """
        Neares Neighbor Algorithm for Job Assignment. From the last node of the tour assigns
        the cheapest job available. Then in the next node assing the cheapest job availabe 
        and so on until the first node
        """
        job = []
        n = len(route)  
        cont = len(route)-1
        while len(job)<n:
            times = {(i,route[cont]):JT[i][route[cont]] for i in range(1,n+1)}
            new = min(times.items(),key=lambda x:x[1]) 
            job.append(new[0][1])
            cont -= 1
        return job

    @staticmethod
    def sort_jobs(route,jobs):
        return [(route[i],jobs[i]) for i in range(len(route))]

    def add_new_constraint(self):
        """
        Add new constraints, builts in this work.
        """
        if self.bounds:
            self.heuristic_jobs = self.NNJA(self.lkh_route[1:],self.JT)
            
            costo_inicial = self.fitness_functions([self.lkh_route[1:],self.heuristic_jobs])[0]
            menor_arco_depot = min(self.TT[0][i] for i in range(1,self.n))

            #Cota superior tiempo de inicio
            for i in self.cities:
                self.modelo.addConstr(self.TS[i]<=costo_inicial)

            #Cota inferior tiempo de inicio
            for i in range(1,self.n):
                self.modelo.addConstr(self.TS[i]>=menor_arco_depot) 

            #Cota superior de solucion inicial (lkh+NNJ)
            self.modelo.addConstr(self.Cmax<=costo_inicial)
            
            self.modelo.update()

    def add_initial_solution(self):
        """
        Add initial solution to MILP from LKH and NNJA
        """

        self.initial_arch = self.__sort_arch(self.lkh_route)
        if self.heuristic_jobs is None:
            self.heuristic_jobs = self.NNJA(self.lkh_route[1:],self.JT) 
        self.initial_job_arch = self.sort_jobs(self.lkh_route[1:],self.heuristic_jobs)
        
        self.modelo.NumStart = 1
        self.modelo.update()
        for s in range(self.modelo.NumStart):
            self.modelo.Params.StartNumber = s
        
            for var in self.modelo.getVars(): 
                if var.VarName[0] == "x":
                    arch_name = tuple(var.varName.split("[")[1][:-1].split(","))
                    if arch_name in self.initial_arch:
                        var.Start = 1

                elif var.VarName[0] == "z":
                    arch_name = tuple(var.varName.split("[")[1][:-1].split(","))
                    if arch_name in self.initial_job_arch:
                        var.Start = 1
        self.modelo.update()

    def optimize(self):
        if self.callback == "None":
            self.modelo._callback_count = 0
            self.modelo._callback_time = 0
            self.modelo.optimize()
        else:
            self.modelo.Params.LazyConstraints = 1
            self.modelo._xvars = self.x
            self.modelo._zvars = self.z
            self.modelo._n = self.n
            self.modelo._callback_count = 0
            self.modelo._callback_time = 0
            self.modelo._epsilon = 0.00001
            self.modelo._DG = nx.DiGraph(nx.complete_graph(self.n))
            self.modelo.optimize(self.callback_dict[self.callback])

    def run(self):
        self.create_base_model()
        self.add_subtour_constraint()
        self.add_new_constraint()
        if self.initial_solution:
            self.add_initial_solution()
        self.optimize()

    def print_results(self):
        dict_status = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}
        lower = self.modelo.ObjBoundC
        objective = float("inf")
        gap =  float("inf")
        if self.modelo.Status == GRB.OPTIMAL or self.modelo.SolCount > 0:
            objective = self.modelo.getObjective().getValue()
            gap = round((objective-lower)/lower*100,4)
            lower = round(lower,2)
            objective = round(objective,2)

        else:
            #print(self.instance, ':Optimization ended with status %d' % self.modelo.Status)
            if self.modelo.SolCount > 0:
                objective = self.modelo.getObjective().getValue()
                lower = self.modelo.getObjective().getValue()
                gap = round((objective-lower)/lower*100,4)
                lower = round(lower,2)
                objective = round(objective,2)
        
        time = round(self.modelo.Runtime,4)
        lower = round(lower,2)
        print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}{:<10}{:<10}".format(self.size,self.instance,objective,lower,gap,time,dict_status[self.modelo.Status],self.modelo.SolCount,self.modelo.NodeCount,self.modelo._callback_count,self.modelo._callback_time))
