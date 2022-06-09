from pyexpat import model
import gurobipy as gp
from gurobipy import *
import pandas as pd
from itertools import combinations
import numpy as np

#gurobi.py tiene menos restricciones, cambios hechos en reunion.
#gurobi.py es lo mismo que cplex.py
#gurobi2.py es lo mismo que cplex2.py

def distancia(i, j):
    global TT
    return TT[(i,j)]

def jobTime(i,j):
    global JT
    #JT[(trabajos,nodos)]
    return JT[(i,j)]

def parametersexcel(excel):
    ruta = "/Users/pablogutierrezaguirre/Desktop/TSPJ_git/Data/instancias_paper/"+excel+".xlsx"
    TT = pd.read_excel(ruta,sheet_name="TT",index_col=0)
    JT = pd.read_excel(ruta,sheet_name="JT",index_col=0)
    #coord = pd.read_excel(excel,sheet_name = "coord",index_col=0,header=0)
    #coord_x = coord["coord_x"]
    #coord_y = coord["coord_y"]
    #print(JT[2][1])
    #JT[NODOS][TRABAJOS]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}

    return nodes,arch,travel_time,job_time

def subtourelim1(modelo, donde):
    # Callback - para usar cortes lazy de eliminación de subtour Eq DFJ1
    n = N
    if donde == GRB.Callback.MIPSOL:
        valoresX = modelo.cbGetSolution(modelo._vars)
        # encontrar ciclo más pequeño
        tour = [i for i in range(n+1)]
        subtour(tour, valoresX,n)
        if len(tour) < n:
            # agregar cortes de elimination de subtour DFJ1
            modelo.cbLazy(gp.quicksum(modelo._vars[i, j]
                                      for i, j in combinations(tour, 2))
                          <= len(tour)-1)

def subtour(subruta, vals,n):
    # obtener una lista con los arcos parte de la solucións
    arcos = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    noVisitados = list(range(n))
    while noVisitados:  # true if list is non-empty
        ciclo = []
        vecinos = noVisitados
        while vecinos:
            actual = vecinos[0]
            ciclo.append(actual)
            noVisitados.remove(actual)
            vecinos = [j for i, j in arcos.select(actual, '*')
                       if j in noVisitados]
        if len(subruta) > len(ciclo):
            subruta[:] = ciclo

def parameterscsv():

    TT = pd.read_csv("/Users/pablogutierrezaguirre/Desktop/TSPJ/TT_paper.csv",index_col= None, header = None)
    JT = pd.read_csv("/Users/pablogutierrezaguirre/Desktop/TSPJ/JT_paper.csv",index_col= None, header = None)

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}
    return nodes,arch,travel_time,job_time

instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
for instancia in instancias[:7]:
    ciudades,arcos,TT,JT = parametersexcel(instancia)
    #ciudades,arcos,TT,JT = parameterscsv()
    dist = {(i, j):distancia(i,j) for i, j in arcos}
    trabajos = ciudades.copy()
    nodos_trabajos = [(i,k) for i in ciudades for k in ciudades]
    jt_aux = np.array(list(JT.values()))
    jt_min = jt_aux[(jt_aux >= 1)].min()

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as modelo:
            # Crear variables
            Cmax = modelo.addVar(name="Cmax")
            x = modelo.addVars(dist.keys(), vtype=GRB.BINARY, name='x')
            z = modelo.addVars(nodos_trabajos, vtype=GRB.BINARY, name='z')
            y = modelo.addVars(arcos,name = "y")
            TS = modelo.addVars(ciudades,name="TS")

            #Crear función objetivo
            modelo.setObjective(Cmax, GRB.MINIMIZE)

            #Esta restricción sin jt_min es equivalente al segundo conjunto de restricciones
            modelo.addConstr(Cmax >= jt_min + quicksum(x[(i,j)]*TT[(i,j)] for i in ciudades for j in ciudades[1:] if i!=j))

            #Restricciones
            for i in ciudades[1:len(ciudades)]:
                modelo.addConstr(Cmax >= TS[i] + quicksum(z[(i,k)]*JT[(k,i)] for k in trabajos if k!=0 ))

            for i in ciudades[1:len(ciudades)]:
                modelo.addConstr(Cmax >= TS[i] + x[(i,0)]*TT[(0,i)])

            for k in trabajos[1:len(ciudades)]:
                modelo.addConstr(quicksum(z[(i,k)] for i in ciudades if i != 0) == 1)

            for i in ciudades[1:len(ciudades)]: 
                modelo.addConstr(quicksum(z[(i,k)] for k in trabajos if k != 0) == 1)
            
            for i in ciudades: #12
                modelo.addConstr(quicksum(x[(i,j)] for j in ciudades if i!=j)==1)

            for j in ciudades: #13
                modelo.addConstr(quicksum(x[(i,j)] for i in ciudades if i!=j)==1) 

            for i in ciudades[1:len(ciudades)]: #14
                modelo.addConstr(quicksum(y[(i,j)] for j in ciudades if i !=j) - quicksum(y[(j,i)] for j in ciudades if i !=j) == 1)

            for i in ciudades[1:len(ciudades)]: #15
                for j in ciudades:
                    if i!=j:
                        modelo.addConstr(y[(i,j)]<= len(ciudades)*x[(i,j)])

            for i in ciudades: #16
                for j in ciudades[1:len(ciudades)]:
                    if i!=j:
                        modelo.addConstr(TS[i] + TT[(j,i)] - (1-x[(i,j)])*30000 <= TS[j])


            
            # Restricción 2-degree
            #modelo.addConstrs(x.sum(i, '*') == 2 for i in range(len(ciudades)))

            # Parámetros
            #modelo.Params.Threads = 6
            #modelo.Params.LazyConstraints = 1
            modelo.setParam('TimeLimit', 999)
            # imprimir modelo
            modelo.optimize()
            #modelo.write("file2.lp")

            # for i in x:
            #     if x[i].X>0.9:
            #         print((round(x[i].X,0),i))   

            lower = modelo.ObjBoundC
            objective = modelo.getObjective().getValue()
            gap = round((objective-lower)/lower*100,4)

            lower = round(modelo.ObjBoundC,4)
            objective = round(modelo.getObjective().getValue(),4)
            time = round(modelo.Runtime,2)
            # instancia, bks, lower, gap, time
            print("{:<10}{:<10}{:<10}{:<10}{:<10}".format(instancia,objective,lower,gap,time))
            #print(f"{instancia} {round(modelo.getObjective().getValue(),1)} {round(modelo.Runtime,2)}")


