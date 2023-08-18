import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import tsplib95
import lkh
import sys
import warnings
import networkx as nx
from typing import Dict, List, Callable, Any
warnings.filterwarnings("ignore")


path = "Codigos/"
class Problem:
    def __init__(self,size,instance):
        self.size = size.lower()
        self.instance = instance
        self.batch = (self.instance-1)//25+1 if self.size != "tsplib" else ""
            
        self.__parameters()
        self.__solve_lkh()
        
    def __parameters(self):
        if self.size == "tsplib":
            location = path+"Data/instancias_paper/"+self.instance+".xlsx"
            TT = pd.read_excel(location,sheet_name="TT",index_col=0)
            JT = pd.read_excel(location,sheet_name="JT",index_col=0)
            self.n = len(list(JT.loc[0].dropna()))

        elif self.size in ("small","medium","large"):
            location = path+"Data/"+str(self.size)+"_problems/Batch_0"+str(self.batch)+"/TSPJ_"+str(self.instance)+self.size[0]
            TT = pd.read_csv(location+"_cost_table_by_coordinates.csv",index_col= None, header = None)
            JT = pd.read_csv(location+"_tasktime_table.csv",index_col= None, header = None)
            self.n = len(list(TT.index.values))

        else:
            TT = pd.read_csv(f"{path}Data/test/1_TT_paper.csv",index_col= None, header = None)
            JT = pd.read_csv(f"{path}Data/test/1_JT_paper.csv",index_col= None, header = None)
            self.n = len(list(TT.index.values))

        self.cities = [i for i in range(self.n)]
        self.arch = [(i,j) for i in self.cities for j in self.cities if i !=j]
        self.TT = {(i,j): TT[i][j] for i,j in self.arch}
        self.JT = {(i,j): JT[i][j] for i in self.cities for j in self.cities}

    def __transform_txt(self,location):
        data = open(location,"r")
        lines = [linea.split() for linea in data]
        data.close()
        lines[0][0] = lines[0][0].replace(","," ")
        lines[0][0] = str(10000000)+" "+ lines[0][0][1:]
        for line in lines[1:-1]:
            line[0] = line[0].replace(",,"," 10000000 ")
            line[0] = line[0].replace(","," ")
        lines[-1][0] = lines[-1][0].replace(","," ")
        lines[-1][0] = lines[-1][0][:-1] +" "+str(10000000)
        
        data = open(path+size+"_"+str(self.batch)+"_"+str(self.instance)+".txt","w")
        data.write("NAME: prueba"+str(self.n)+"\n")
        data.write("TYPE: TSP\n")
        data.write(f"COMMENT: {self.n} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n")
        data.write(f"DIMENSION: {self.n}\n")
        data.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n")
        for i in lines:
            data.write(i[0]+"\n")
        data.close()

    def __solve_lkh(self):
        if self.size in ("small","medium","large"):
            instance_location = path+"Data/"+str(self.size)+"_problems/Batch_0"+str(self.batch)+"/TSPJ_"+str(self.instance)+self.size[0]+"_cost_table_by_coordinates.csv"
        elif self.size == "tsplib":
            instance_location = path+f"Data/Tsplib_problems/TT_{self.instance}.csv"
        else:
            self.lkh_route = [i for i in range(self.n)]
            return
        self.__transform_txt(instance_location)
        
        problem = tsplib95.load(path+self.size+"_"+str(self.batch)+"_"+str(self.instance)+".txt")
        os.remove(path+self.size+"_"+str(self.batch)+"_"+str(self.instance)+".txt")

        solver_path = path+'LKH-3.0.7/LKH'

        ciudad = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)[0]
        self.lkh_route = [i-1 for i in ciudad]

    def get_lkh_route(self):
        return self.lkh_route
    
    def fitness_functions(self,solution):
        ciudad = solution[0]
        trabajo = solution[1]
        n = len(ciudad)
        suma_ac = [self.TT[(0, ciudad[0])]]
        suma = suma_ac[-1]
        maxtime = suma_ac[-1]+self.JT[(trabajo[0],ciudad[0])] 
        cmax = 0
        i = 0
        while i < n -1:
            suma += self.TT[(ciudad[i], ciudad[i + 1])]
            suma_ac.append(suma)
            maxtime = suma_ac[-1] + self.JT[(trabajo[i+1],ciudad[i+1])  ]
            if maxtime > cmax:
                cmax = maxtime
            i += 1
        suma += self.TT[(ciudad[-1], 0)]
        suma_ac.append(suma)
        if suma_ac[-1]>cmax:
            cmax = suma_ac[-1]
        return cmax,

class ModeloMatematico(Problem):
    def __init__(self,size:str,
                 instance,
                 output = False,
                 subtour : str = "GG",
                 initial_solution : bool = True,
                 callback : Callable = None,
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
        self.output = output
        self.subtour = subtour
        self.initial_solution = initial_solution
        self.callback = callback
        self.bounds = bounds

        self.compute_M()
        self.jobs = self.cities.copy()
        self.jobs_arch = [(i,k) for i in self.cities for k in self.cities]
        
        jt_aux = np.array(list(self.JT.values()))
        self.jt_min = jt_aux[(jt_aux >= 1)].min()

    def compute_M(self):
        """
        This method computes the M value, it can be overwritting.
        """
        self.M = 0
        for i in range(self.n):
            max_t = 0
            max_tt = 0
            for j in range(self.n):
                if i!=j and self.TT[(i,j)]>max_t:
                    max_t = self.TT[(i,j)]
                
                if j!= 0 and self.JT[(j,i)]>max_tt:
                    max_tt = self.JT[(j,i)]
            self.M += max_t #+max_tt

    def create_base_model(self):
        """
        Create the initial MILP, whithout any additional constraint.
        """
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if output else 0)
        env.start()
        self.modelo = gp.Model(env=env)

        self.Cmax = self.modelo.addVar(vtype=GRB.CONTINUOUS,name="Cmax")

        self.x = self.modelo.addVars(self.TT.keys(), vtype=GRB.BINARY, name='x')
        self.z = self.modelo.addVars(self.jobs_arch, vtype=GRB.BINARY, name='z')

        self.TS = self.modelo.addVars(self.cities,vtype=GRB.CONTINUOUS,name="TS")

        self.modelo.setObjective(self.Cmax, GRB.MINIMIZE)

        #Restricción adicional de reforzamiento
        self.modelo.addConstr(self.Cmax >= self.jt_min + gp.quicksum(self.x[(i,j)]*self.TT[(i,j)] for i in self.cities for j in self.cities[1:] if i!=j))

        self.modelo.addConstrs(self.x.sum(i,'*') == 1 for i in self.cities) # Outgoing
        self.modelo.addConstrs(self.x.sum('*', j) == 1 for j in self.cities) # Incoming 

        #Restricciones
        for i in self.cities[1:self.n]:
            self.modelo.addConstr(self.Cmax >= self.TS[i] + gp.quicksum(self.z[(i,k)]*self.JT[(k,i)] for k in self.jobs if k!=0 ))

        for i in self.cities[1:self.n]:
            self.modelo.addConstr(self.Cmax >= self.TS[i] + self.x[(i,0)]*self.TT[(0,i)])

        for k in self.jobs[1:self.n]:
            self.modelo.addConstr(gp.quicksum(self.z[(i,k)] for i in self.cities if i != 0) == 1)

        for i in self.cities[1:self.n]: 
            self.modelo.addConstr(gp.quicksum(self.z[(i,k)] for k in self.jobs if k != 0) == 1)
        

        for i in self.cities: #16
            for j in self.cities[1:self.n]:
                if i!=j:
                    self.modelo.addConstr(self.TS[i] + self.TT[(j,i)] - (1-self.x[(i,j)])*self.M <= self.TS[j])

        
        self.modelo.Params.Threads = 1
        self.modelo.Params.TimeLimit = 1500
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
            times = {(i,route[cont]):JT[(i,route[cont])] for i in range(1,n+1)}
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
            menor_arco_depot = min(self.TT[(0,i)] for i in range(1,self.n))

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
        if self.callback == None:
            self.modelo._callback_count = 0
            self.modelo.optimize()
        else:
            self.modelo.Params.LazyConstraints = 1
            self.modelo._xvars = self.x
            self.modelo._zvars = self.z
            self.modelo._n = self.n
            self.modelo._callback_count = 0
            self.modelo._epsilon = 0.00001
            self.modelo._DG = nx.DiGraph(nx.complete_graph(self.n))
            self.modelo.optimize(self.callback)

    def run(self):
        self.create_base_model()
        self.add_subtour_constraint()
        self.add_new_constraint()
        if self.initial_solution:
            self.add_initial_solution()
        self.optimize()

    def print_model(self):
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
        print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}{:<10}".format(self.instance,objective,lower,gap,time,dict_status[self.modelo.Status],self.modelo.SolCount,self.modelo.NodeCount,self.modelo._callback_count))

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
bounds = None
#Considerar con y sin cotas

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

for i in range(len(opts)):
    if opts[i][0][1:] == "tipo":  size  = (opts[i][1])
    elif   opts[i][0][1:] == "subtour": subtour = int(opts[i][1])
    elif opts[i][0][1:] == "instancia":
        try:
            instance = int(opts[i][1])  
        except ValueError:
            instance = opts[i][1]  
    elif   opts[i][0][1:] == "solinicial": initial_solution = True if str(opts[i][1]) =="True" else False
    elif   opts[i][0][1:] == "output": output = True if str(opts[i][1]) =="True" else False

instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
configs = ("BASE","GG","mycallback","dfj","cut")
for i,cb in enumerate((1,2,subtourelim,DFJ_fractional_callback,CUT_fractional_callback)):
    print(configs[i]+":")
    for instance in instancias:
        MM = ModeloMatematico(size,instance)
        MM.callback = cb
        if cb == 1:
            MM.callback = None
            MM.subtour = "WC"
        elif cb == 2:
            MM.callback = None
            MM.subtour = "GG"
        else:
            MM.callback = cb
            MM.subtour = "WC"
        MM.output = output
        MM.initial_solution = initial_solution
        MM.run()
        MM.print_model()
    print("")    




