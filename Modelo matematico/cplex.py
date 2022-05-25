import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
from docplex.mp.model import Model
os.system("clear")

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

def main():
    nodos,arcos,TT,JT= parameters("/Users/pablogutierrezaguirre/Desktop/TSPJ/Modelo matematico/TSPJ_gr17.xlsx")
    #nodos,arcos,TT,JT= parameterscsv()
    trabajos = nodos.copy()
    nodos_trabajos = [(i,k) for i in nodos for k in trabajos]
    n = len(nodos)

    mdl=Model("Modelo")
    #Variables
    Cmax = mdl.integer_var(name = "Cmax")
    x = mdl.binary_var_dict(arcos,name="x")
    # y = mdl.continuous_var_dict(arcos, name = "y")
    z = mdl.binary_var_dict(nodos_trabajos,name = "z")
    TS = mdl.continuous_var_list(nodos, name ="TS")

    #Funcion objetivo
    mdl.minimize(Cmax)

    #Restricciones
    for i in nodos[1:len(nodos)]:
        mdl.add_constraint(Cmax >= TS[i] + mdl.sum(z[(i,k)]*JT[(k,i)] for k in trabajos if k!=0 ))

    for i in nodos[1:len(nodos)]:
        mdl.add_constraint(Cmax >= TS[i] + x[(i,0)]*TT[(0,i)])

    for k in trabajos[1:len(nodos)]:
        mdl.add_constraint(mdl.sum(z[(i,k)] for i in nodos if i != 0) == 1)

    for i in nodos[1:len(nodos)]: 
        mdl.add_constraint(mdl.sum(z[(i,k)] for k in trabajos if k != 0) == 1)
    
    for i in nodos: #12
        mdl.add_constraint(mdl.sum(x[(i,j)] for j in nodos if i!=j)==1)

    for j in nodos: #13
        mdl.add_constraint(mdl.sum(x[(i,j)] for i in nodos if i!=j)==1)

    # for i in nodos[1:len(nodos)]: #14
    #     mdl.add_constraint(mdl.sum(y[(i,j)] for j in nodos if i !=j) - mdl.sum(y[(j,i)] for j in nodos if i !=j) == 1)

    # for i in nodos[1:len(nodos)]: #15
    #     for j in nodos:
    #         if i!=j:
    #             mdl.add_constraint(y[(i,j)]<= n*x[(i,j)])

    for i in nodos: #16
        for j in nodos[1:len(nodos)]:
            if i!=j:
                mdl.add_constraint(TS[i] + TT[(j,i)] - (1-x[(i,j)])*300000 <= TS[j])

    print(mdl.export_to_string())
    solucion=mdl.solve(log_output=True)
    print(mdl.get_solve_status())
    solucion.display()

main()
