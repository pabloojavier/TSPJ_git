import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
from docplex.mp.model import Model
#os.system("clear")

#gurobi.py tiene menos restricciones, cambios hechos en reunion.
#gurobi.py es lo mismo que cplex.py
#gurobi2.py es lo mismo que cplex2.py

def parameters(excel):
    TT = pd.read_excel(excel,sheet_name="TT",index_col=0)
    JT = pd.read_excel(excel,sheet_name="JT",index_col=0)
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

def parameterscsv():

    TT = pd.read_csv("Data/test/TT_paper.csv",index_col= None, header = None)
    JT = pd.read_csv("Data/test/JT_paper.csv",index_col= None, header = None)

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}
    return nodes,arch,travel_time,job_time

def leer_solucion(x,z,n):
    ruta = [i for i in x if x[i].solution_value>0.8]
    ruta = ordenar_ruta(ruta,n)[1:]

    trabajos = [i for i in z if z[i].solution_value>0.8]
    trabajos = ordenar_trabajos(ruta,trabajos,n)

    return [ruta,trabajos]

def ordenar_ruta(lista,n):
    solucion = []
    solucion.append(0)
    while len(solucion)!= n:
        for i in lista:
            if i[0]==solucion[-1]:
                solucion.append(i[1])
    return solucion

def ordenar_trabajos(ruta,lista,n):
    trabajos = []
    for i in ruta:
        for j in lista:
            if j[0]==i:
                trabajos.append(j[1])
    return trabajos

def main(instancia):
    nodos,arcos,TT,JT= parameters("Data/instancias_paper/"+instancia+".xlsx")
    #nodos,arcos,TT,JT= parameterscsv()
    trabajos = nodos.copy()
    nodos_trabajos = [(i,k) for i in nodos for k in trabajos]
    n = len(nodos)

    mdl=Model("Modelo")
    #Variables
    Cmax = mdl.continuous_var(name = "Cmax")
    x = mdl.binary_var_dict(arcos,name="x")
    y = mdl.continuous_var_dict(arcos, name = "y")
    z = mdl.binary_var_dict(nodos_trabajos,name = "z")
    TS = mdl.continuous_var_list(nodos, name ="TS")

    jt_aux = np.array(list(JT.values()))
    jt_min = jt_aux[(jt_aux >= 1)].min()
    #Funcion objetivo
    mdl.minimize(Cmax)

    #Restricciones
    mdl.add_constraint(Cmax >= jt_min + mdl.sum(x[(i,j)]*TT[(i,j)] for i in nodos for j in nodos[1:] if i!=j))

    for i in nodos[1:len(nodos)]: #8
        mdl.add_constraint(Cmax >= TS[i] + mdl.sum(z[(i,k)]*JT[(k,i)] for k in trabajos if k!=0 ))

    for i in nodos[1:len(nodos)]: #9
        mdl.add_constraint(Cmax >= TS[i] + x[(i,0)]*TT[(0,i)])

    for k in trabajos[1:len(nodos)]: #10
        mdl.add_constraint(mdl.sum(z[(i,k)] for i in nodos if i != 0) == 1)

    for i in nodos[1:len(nodos)]: #11
        mdl.add_constraint(mdl.sum(z[(i,k)] for k in trabajos if k != 0) == 1)
    
    for i in nodos: #12
        mdl.add_constraint(mdl.sum(x[(i,j)] for j in nodos if i!=j)==1)

    for j in nodos: #13
        mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodos if i!=j)==1)

    for i in nodos[1:len(nodos)]: #14
        mdl.add_constraint(mdl.sum(y[(i,j)] for j in nodos if i !=j) - mdl.sum(y[(j,i)] for j in nodos if i !=j) == 1)

    for i in nodos[1:len(nodos)]: #15
        for j in nodos:
            if i!=j:
                mdl.add_constraint(y[(i,j)]<= n*x[(i,j)])

    for i in nodos: #16
        for j in nodos[1:len(nodos)]:
            if i!=j:
                mdl.add_constraint(TS[i] + TT[(j,i)] - (1-x[(i,j)])*30000 <= TS[j])


    #print(mdl.export_to_string())
    mdl.set_time_limit(999)
    mdl.solve(log_output=False)
    #print(mdl.get_solve_status())
    #solucion.display()
    print("{:<10}{:<10}{:<10}".format(instancia,round(mdl.objective_value,1),round(mdl.solve_details.time,2)))

instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
for i in instancias:
    main(i)


# gr17      2760.0    0.71      
# gr21      7788.0    0.79      
# gr24      1806.0    1.23      
# fri26     1283.0    1.61      
# bays29    2916.0    4.84      
# gr48      7282.0    36.89     
# eil51     628.5     24.22     
# berlin52  11087.2   32.32     
