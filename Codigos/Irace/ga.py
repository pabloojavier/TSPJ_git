#!/usr/bin/env python3.9
from operator import index
from numpy.random.mtrand import rand, seed
import matplotlib.pyplot as plt
import random
import time
#import animated_visualizer as animacion
import pandas as pd
import os
import array
import sys
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from itertools import combinations

import gurobipy as gp
from gurobipy import *

import requests
import tsplib95
import lkh


#path = "Codigos/"
path="../"
#path = ""
flagRutas = False
ciudad_gurobi = []
#os.system("clear")

def OX(ind1, ind2):
    """Executes an ordered crossover (OX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates holes in the input
    individuals. A hole is created when an attribute of an individual is
    between the two crossover points of the other individual. Then it rotates
    the element so that all holes are between the crossover points and fills
    them with the removed elements in order. For more details see
    [Goldberg1989]_.

    This function uses the :func:`~random.sample` function from the python base
    :mod:`random` module.

    .. [Goldberg1989] Goldberg. Genetic algorithms in search,
       optimization and machine learning. Addison Wesley, 1989
    """
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)

    for i in range(size):
        ind1[i],ind2[i] = ind1[i]-1,ind2[i]-1

    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if (i < a or i > b):
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False


    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]


    #return ind1, ind2
    for i in range(size): 
        ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

def PMX(ind1, ind2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        ind1[i],ind2[i] = ind1[i]-1,ind2[i]-1


    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    #return ind1, ind2
    for i in range(size): 
        ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

def UMPX(ind1, ind2, indpb=0.5):
    """Executes a uniform partially matched crossover (UPMX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values chosen at random with a probability of *indpb* in the two
    parents and swapping the values of those indexes. For more details see
    [Cicirello2000]_.

    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.

    .. [Cicirello2000] Cicirello and Smith, "Modeling GA performance for
       control parameter optimization", 2000.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        ind1[i],ind2[i] = ind1[i]-1,ind2[i]-1
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(size):
        if random.random() < indpb:
            # Keep track of the selected values
            temp1 = ind1[i]
            temp2 = ind2[i]
            # Swap the matched value
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    #return ind1, ind2
    for i in range(size):
        ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

def cruzamiento(ind1,ind2):
    value = random.uniform(0,1)
    if value <P_OX:
        OX(ind1,ind2)
    elif value <P_OX+P_PMX:
        PMX(ind1,ind2)
    elif value<P_OX+P_PMX+P_UMPX:
        UMPX(ind1,ind2)

def parametersexcel(excel):

    ruta = path+"Data/instancias_paper/"+excel+".xlsx"
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

def parameterscsvpaper():

    TT = pd.read_csv(f"{path}Data/test/TT2_paper.csv",index_col= None, header = None)
    JT = pd.read_csv(f"{path}Data/test/JT2_paper.csv",index_col= None, header = None)

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}
    return nodes,arch,travel_time,job_time

def parameters(size,batch,instancia):
    
    nameTT = path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_cost_table_by_coordinates.csv"
    nameJT = path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_tasktime_table.csv"
    #namecoord =path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_nodes_table_by_coordinates.csv"

    TT = pd.read_csv(nameTT,index_col= None, header = None)
    JT = pd.read_csv(nameJT,index_col= None, header = None)
    #coord = pd.read_csv(namecoord,index_col=None,header=None)

    #coord_x = coord[0]
    #coord_y = coord[1]

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}

    return nodes,arch,travel_time,job_time

def distancia(i, j):
    global TT
    return TT[(i,j)]

def costoTotal(ciudad2):
    #ciudad2 = [[20, 1, 19, 9, 3, 14, 17, 16, 13, 21, 10, 18, 24, 6, 22, 26, 15, 12, 23, 7, 27, 5, 11, 8, 4, 25, 28, 2], [7, 28, 27, 2, 25, 5, 3, 26, 13, 20, 6, 24, 19, 15, 14, 12, 16, 23, 4, 10, 8, 11, 21, 22, 18, 17, 1, 9]]
    #ciudad2 = [[3, 11, 23, 4, 20, 7, 6, 5, 15, 10, 2, 16, 9, 13, 19, 1, 14, 18, 21, 17, 22, 8, 12], [16, 4, 23, 10, 2, 12, 17, 21, 20, 1, 5, 11, 19, 7, 22, 9, 18, 6, 13, 3, 14, 8, 15]]
    #ciudad2 = [[43, 40, 33, 22, 24, 2, 18, 3, 29, 37, 19, 34, 41, 38, 39, 36, 1, 44, 42, 46, 17, 45, 27, 6, 28, 12, 47, 15, 10, 35, 5, 25, 14, 23, 9, 11, 30, 4, 32, 7, 21, 8, 13, 20, 31, 26, 16], [45, 36, 38, 20, 24, 40, 12, 27, 32, 7, 19, 14, 4, 3, 22, 1, 28, 13, 47, 43, 16, 9, 5, 11, 18, 23, 35, 10, 39, 29, 33, 46, 15, 21, 17, 25, 6, 31, 44, 34, 26, 37, 8, 42, 30, 41, 2]]
    #ciudad2= [[1,2,4,8,6,12,21,39,36,37,35,32,38,33,25,19,26,29,24,18,16,7,3,13,22,30,27,34,31,28,20,17,15,9,10,5,11,14,23,40],[22,40,20,33,6,5,24,25,17,16,36,11,27,14,3,34,21,9,12,35,4,18,28,7,19,38,13,39,32,15,2,26,1,10,30,37,31,8,29,23]]
    
    ciudad = ciudad2[0]
    trabajo = ciudad2[1]
    n = len(ciudad)
    suma_ac = [distancia(0, ciudad[0])]
    suma = suma_ac[-1]
    maxtime = suma_ac[-1]+jobTime(trabajo[0],ciudad[0]) 
    cmax = 0
    i = 0
    while i < n -1:
        suma += distancia(ciudad[i], ciudad[i + 1])
        suma_ac.append(suma)
        maxtime = suma_ac[-1] + jobTime(trabajo[i+1],ciudad[i+1])  
        #print(maxtime)
        #print("costo: ",suma_ac[-1],maxtime)
        #print(suma_ac,maxtime)
        if maxtime > cmax:
            cmax = maxtime
            #print("COSTO TOTAL: ",ciudad,ciudad[i+1])
        i += 1
    suma += distancia(ciudad[-1], 0)
    suma_ac.append(suma)
    if suma_ac[-1]>cmax:
        cmax = suma_ac[-1]
        #print("costo: ",suma_ac[-1],maxtime)
        #print("COSTO TOTAL: ",ciudad[-1])
    
    #print("cmas: ",cmax)
    # print("ac",suma_ac)
    #exit(0)

    return cmax,

def jobTime(i,j):
    global JT
    #JT[(trabajos,nodos)]
    return JT[(i,j)]

def vecinoMasCercano(n):
    desde = random.randint(1,n)
    actual = desde
    ciudad = []
    ciudad.append(desde)
    seleccionada = [False] * n
    seleccionada[actual-1] = True
    # print(seleccionada)
    while len(ciudad) < n:
        min = 9999999
        for candidata in range(1,n+1):
            if seleccionada[candidata-1] == False and candidata != actual:
                costo = distancia(actual, candidata)
                if costo < min:
                    min = costo
                    siguiente = candidata

        ciudad.append(siguiente)
        seleccionada[siguiente-1] = True
        actual = siguiente
    return ciudad

def transformar_txt1(ruta,n):
    archivo = open(ruta,"r")
    lineas = [linea.split() for linea in archivo]
    archivo.close()
    lineas[0][0] = lineas[0][0].replace(","," ")
    lineas[0][0] = str(10000000)+" "+ lineas[0][0][1:]
    for i in lineas[1:-1]:
        i[0] = i[0].replace(",,"," 10000000 ")
        i[0] = i[0].replace(","," ")
    lineas[-1][0] = lineas[-1][0].replace(","," ")
    lineas[-1][0] = lineas[-1][0][:-1] +" "+str(10000000)
    
    #archivo = open(path+size+"_"+str(batch)+"_"+str(instancia)+"_"+str(semilla)+".txt","w")
    # archivo.write("NAME: prueba"+str(n+1)+"\n")
    # archivo.write("TYPE: TSP\n")
    # archivo.write(f"COMMENT: {n+1} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n")
    # archivo.write(f"DIMENSION: {n+1}\n")
    # archivo.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n")
    # for i in lineas:
    #     archivo.write(i[0]+"\n")
    #archivo.close()
    string_problem  = "NAME: prueba"+str(n+1)+"\n"
    string_problem += "TYPE: TSP\n"
    string_problem += f"COMMENT: {n+1} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n"
    string_problem += f"DIMENSION: {n+1}\n"
    string_problem += f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n"
    for i in lineas:
        string_problem += i[0]+"\n"
    return string_problem

def transformar_txt2(ruta):
    datos = pd.read_excel(ruta,sheet_name="TT",index_col=0)
    n = len(datos[0])
    # archivo = open(path+size+"_"+str(instancia)+"_"+str(semilla)+".txt","w")
    # archivo.write("NAME: tsplib"+str(n)+"\n")
    # archivo.write("TYPE: TSP\n")
    # archivo.write(f"COMMENT: {n} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n")
    # archivo.write(f"DIMENSION: {n}\n")
    # archivo.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n")
    # archivo.write(datos.replace(0,10000000).to_csv(header=None,index=False).replace(","," "))
    # archivo.close()

    string_problem  = "NAME: prueba"+str(n)+"\n"
    string_problem += "TYPE: TSP\n"
    string_problem += f"COMMENT: {n} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n"
    string_problem += f"DIMENSION: {n}\n"
    string_problem += f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n"
    string_problem += datos.replace(0,10000000).to_csv(header=None,index=False).replace(","," ")
    return string_problem

def costo_ciudad(ciudad,n):
    suma_ac = [distancia(0, ciudad[0])]
    suma = suma_ac[-1]
    i = 0
    while i < n -1:
        suma += distancia(ciudad[i], ciudad[i + 1]) 
        #print("costo: ",suma_ac[-1],maxtime)
        #print(suma_ac,maxtime)
        i += 1
    suma += distancia(ciudad[-1], 0)
    return suma

def solve_lkh(n):
    if isinstance(instancia,int):
        ruta_instancia = path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_cost_table_by_coordinates.csv"
        string_problem = transformar_txt1(ruta_instancia,n)
        problem = tsplib95.parse(string_problem)
    else:
        ruta_instancia = path+"Data/instancias_paper/"+instancia+".xlsx"
        string_problem = transformar_txt2(ruta_instancia)
        problem = tsplib95.parse(string_problem)

    #problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
    solver_path = path+'LKH-3.0.7/LKH'
    ciudad = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)[0]
    ciudad = [i-1 for i in ciudad if i != 1]
    del string_problem
    return ciudad

def generarRuta(n):
    global flagRutas
    global ciudad_gurobi
    
    # ciudad = vecinoMasCercano(n)
    # print("VMC",costo_ciudad(ciudad,n))

    # ciudad = solve_lkh(n)
    # print("LKH",costo_ciudad(ciudad,n))
    
    # ciudad = tsp_gurobi_cortes(n,180)
    # print("gurobi",costo_ciudad(ciudad,n))
    # exit(0)

    aleatorio =random.uniform(0, 1) 

    if aleatorio < P_NNH:
        ciudad = vecinoMasCercano(n)

    elif aleatorio <P_NNH+P_TSP:
        if flagRutas==False:
            if n<150:
                ciudad = tsp_gurobi_cortes(n,2000)
            else:
                ciudad = solve_lkh(n)
            ciudad_gurobi=ciudad.copy()
            flagRutas=True
        else:
            ciudad = ciudad_gurobi
    else:
        ciudad = [i for i in range(1, n+1)]
        random.shuffle(ciudad)
    return ciudad

def ordenar_solucion(lista,n):
    actual = 0
    _solucion = []
    while len(_solucion) < n+1:
        _solucion.append(actual)
        for i in lista:
            if i[0]==actual and i[1] not in _solucion:
                siguiente = i[1]
                actual = siguiente
                break
    return _solucion

def print_parameters(name,value):
    if multi =="False":
        print(f"{name} = {value}")

def tsp_gurobi_gg(n,tiempo=10):
    ciudades = [i for i in range(n+1)]
    arcos = [(i, j) for i in ciudades for j in ciudades if i != j]
    dist = {(i, j):distancia(i,j) for i, j in arcos}
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as modelo:
            x = tupledict()
            #Variable y funcion objetivo
            for i, j in dist.keys():
                x[i,j] = modelo.addVar(obj = dist[i,j], vtype = GRB.BINARY,name = "e[%d,%d]" % (i,j))

            # Restricciones de salida 
            modelo.addConstrs(gp.quicksum(x[i, j] for j in ciudades if j != i) == 1 for i in ciudades)
            # Restricciones de entrada 
            modelo.addConstrs(gp.quicksum(x[i, j] for i in ciudades if i != j) == 1 for j in ciudades)

            g = modelo.addVars(arcos, vtype=GRB.CONTINUOUS, name = 'g')
            modelo.addConstr(gp.quicksum(g[i,j] for i,j in arcos if i == 0) == 0, "gg_inicio_%d" % 0)
            for c in ciudades:
                if c != 0:
                    modelo.addConstr(((gp.quicksum(g[i,j] for i,j in arcos if i==c)) - (gp.quicksum(g[j,i] for i,j in arcos if i==c))) == 1, "gg_%d" % c)

            for i,j in arcos:
                modelo.addConstr(g[i,j] >= 0, "gg_lb_%d_%d" % (i,j))

            for i,j in arcos:
                modelo.addConstr(g[i,j] <= n * x[i,j], "gg_ub_%d_%d" % (i,j))


            modelo.Params.Threads = 4
            #Tiempo limite
            modelo.setParam('TimeLimit', tiempo)

            # imprimir modelo
            # Resolver
            modelo.optimize()
            solucion = []
            for i in x:
                if x[i].X ==1:
                    solucion.append(i)
            solucion = ordenar_solucion(solucion)
            #print("Costo  : %g" % modelo.ObjVal)
            #print("Tiempo : %s" % str(modelo.Runtime+time.time()-inicio))
    return solucion[1:]

def tsp_gurobi_cortes(n,tiempo=10):
    N = n

    ciudades = [i for i in range(n+1)]
    arcos = [(i, j) for i in ciudades for j in ciudades if i < j]
    dist = {(i, j):distancia(i,j) for i, j in arcos}

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as modelo:
            # Crear variables y función objectivo
            x = modelo.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

            # como es simétrico copiamos sus opuestos
            for i, j in x.keys():
                x[j, i] = x[i, j]

            # Restricción 2-degree
            modelo.addConstrs(x.sum(i, '*') == 2 for i in range(n+1))

            # Parámetros
            modelo.Params.Threads = 1
            modelo._vars = x
            modelo.Params.LazyConstraints = 1
            modelo.setParam('TimeLimit', tiempo)
            # imprimir modelo
            modelo.optimize(subtourelim1)
            solucion = []
            for i in x:
                if x[i].X >0.8:
                    solucion.append(i)
            solucion = ordenar_solucion(solucion,n)
    return solucion[1:]

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

def creador_trabajos(n):
    pass

def crear_trabajos2(ruta,trabajo):
    value = random.uniform(0,1)

    if value < 1-P_NNHJ:
        n = len(ruta)
        trabajo = [i for i in range(1,n+1)]
        random.shuffle(trabajo)

    else:
        #print("entré")
        ciudad = ruta
        trabajo = []
        n = len(ciudad)  
        cont = len(ruta)-1
        #QUE RECORRA HACIA ATRÁS PARTIENDO DE DISTINTOS NODOS
        while len(trabajo)<n:
            tiempos = {(i,ciudad[cont]):jobTime(i,ciudad[cont]) for i in range(1,n+1)}
            nuevo = min(tiempos.items(),key=lambda x:x[1]) 
            trabajo.append(nuevo[0][1])
            cont -= 1
    return trabajo

def DosOpt(ciudad):
    ciudad.insert(0,0)
    actual = 0
    n = len(ciudad)
    flag = True
    contar = 0
    # k = random.randint(0, len(ciudad) - 1)
    # ciudad = ciudad[k:] + ciudad[:k]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            nuevoCosto = distancia(ciudad[i], ciudad[j]) + distancia(ciudad[i + 1], ciudad[j + 1]) - distancia(ciudad[i], ciudad[i + 1]) - distancia(ciudad[j], ciudad[j + 1])
            if nuevoCosto < actual:
                actual = nuevoCosto
                min_i, min_j = i, j
                # Al primer cambio se sale
                contar += 1
                if contar == 1 :
                    flag = False

        if flag == False:
            break

    # Actualiza la subruta se encontró
    if actual < 0:
        ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]
    cero = ciudad.index(0)
    if cero !=0:
        ciudad = ciudad[cero:len(ciudad)]+ciudad[0:cero]
    ciudad.remove(0)

def DosOpt_v2(ciudad,trabajos):
    """
    Calcular costo total directamente acá (sin función) para obtener la posición del Cmax (o X primeros cmax),
    con esto, al calcular un nuevo costo se analizarán ambos nodos cambiados y se comparan con los cmax
    anteriores, así, 
    MISMA LOGICA QUE BUSQUEDA LOCAL PARA LOS TRABAJOS
    """
    #ciudad.insert(0,0)
    actual = costoTotal([ciudad,trabajos])[0]
    n = len(ciudad)
    flag = True
    contar = 0
    # k = random.randint(0, len(ciudad) - 1)
    # ciudad = ciudad[k:] + ciudad[:k]
    min_i,min_j = 0,0
    for i in range(n - 2):
        for j in range(i + 1, n - 2):
            guardar = ciudad.copy()
            ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]
            #nuevoCosto1 = costoTotal([ciudad,trabajos])[0]
            nuevoCosto= distancia(ciudad[i], ciudad[j]) + distancia(ciudad[i + 1], ciudad[j + 1]) - distancia(ciudad[i], ciudad[i + 1]) - distancia(ciudad[j], ciudad[j + 1])

            if nuevoCosto < 0:
                nuevoCosto = costoTotal([ciudad,trabajos])[0]
                if nuevoCosto< actual:
                    actual = nuevoCosto
                    min_i, min_j = i, j
                    # Al primer cambio se sale
                    contar += 1
                    if contar == 1 :
                        flag = False
            else:
                ciudad = guardar.copy()

        if flag == False:
            break

    # Actualiza la subruta se encontró
    if actual < 0:
        print("hola")
        ciudad[min_i + 1 : min_j + 1] = ciudad[min_i + 1 : min_j + 1][::-1]
    """
    # cero = ciudad.index(0)
    # if cero !=0:
    #     ciudad = ciudad[cero:len(ciudad)]+ciudad[0:cero]
    # ciudad.remove(0)
    """

def DosOpt_v3(ciudad,trabajo):
    # print("ENTRÉ")
    # a=[[1, 2, 3, 4, 5], [5, 3, 4, 1, 2]]
    # ciudad = a[0]
    # trabajo = a[1]

    n = len(ciudad)
    suma_ac = [distancia(0, ciudad[0])]
    #print(distancia(0, ciudad[0]),0,ciudad[0])
    suma = suma_ac[-1]
    maxtime = suma_ac[-1]+jobTime(trabajo[0],ciudad[0]) 
    #cmax1 = [0]
    #i1_guardado = [0]
    cmax_antiguo = 0
    cmax2_antiguo = 0
    best_i = 0
    i2 = 0
    i = 0
    while i < n - 1:
        suma += distancia(ciudad[i], ciudad[i + 1])
        #print(distancia(ciudad[i], ciudad[i + 1]),ciudad[i], ciudad[i + 1])
        suma_ac.append(suma)
        maxtime = suma_ac[-1] + jobTime(trabajo[i+1],ciudad[i+1])  
        #print("costo2: ",suma_ac[-1],maxtime)
        if maxtime > cmax_antiguo:
            cmax2_antiguo = cmax_antiguo
            i2 = best_i
            cmax_antiguo = maxtime
            best_i = i+1
        elif maxtime > cmax2_antiguo:
            cmax2_antiguo = maxtime
            i2 = i

        i += 1
    suma += distancia(ciudad[-1], 0)
    suma_ac.append(suma)
    if suma_ac[-1]>cmax_antiguo:
        cmax2_antiguo = cmax_antiguo
        i2 = best_i
        best_i = n
        cmax1 = suma_ac[-1]
    
    #print("trabajo: ",trabajo)
    aux = -1
    for i in range(len(trabajo)):
        if i != best_i:
            cmax1_nuevo = suma_ac[best_i] + jobTime(trabajo[i],ciudad[best_i])
            cambio2 = suma_ac[i] + jobTime(trabajo[i],ciudad[i])
            #print("i: ",trabajo[i],trabajo[best_i])
            #print("nuevos: ",cmax1_nuevo,cambio2)
            if cmax1_nuevo < cmax_antiguo:
                if cmax1_nuevo > cambio2:
                    if cmax1_nuevo < cmax2_antiguo: 
                        aux = cmax2_antiguo
                        ciudad[best_i],ciudad[i] = ciudad[i],ciudad[best_i]
                        #print("entré 1")
                        break
                    else:
                        ciudad[best_i],ciudad[i] = ciudad[i],ciudad[best_i]
                        #print("entré 2")
                        aux = cmax1_nuevo
                        break
                else: #cmax nuevo mayor o igual al segundo mas grande 
                    if cambio2>cmax2_antiguo:
                        ciudad[best_i],ciudad[i] = ciudad[i],ciudad[best_i]
                        #print("entré 3")
                        aux = cambio2
                        break
                    else:
                        aux = cmax2_antiguo
                        #print("entré 4")
                        ciudad[best_i],ciudad[i] = ciudad[i],ciudad[best_i]
                        break
    """
    #print(cmaxs)
    #print("entré2 ",ciudad,trabajo)
    # aux3, = costoTotal([ciudad,trabajo])
    # if aux3 != aux and aux != -1:
    #     print("cmax: ",aux3,aux)
    #     print("ERROR")
    #     exit(0)
    """

def perturbation(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp

def perturbation2(ciudad):
    i = 0
    j = 0
    n = len(ciudad)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    ciudad[i : j] = ciudad[i : j][::-1]

def perturbation3(ciudad):
    i = 0
    j = 0
    n = len(ciudad)

    while i == j:
        i = random.randint(0, n - 2)
        # j = random.randint(0, n - 1)
    j = i + 1
        # intercambio
    temp = ciudad[i]
    ciudad[i] = ciudad[j]
    ciudad[j] = temp

def mutSet(ciudad,trabajos):
    value2 = random.uniform(0,1)
    if value2<MS1:
        value = random.uniform(0, 1)
        if value < P_EM:
            perturbation(ciudad)
        elif value < P_EM+P_RM:
            perturbation2(ciudad)
        elif value <P_EM+P_RM+P_SM:
            perturbation3(ciudad)
        elif value <P_EM+P_RM+P_SM+P_2OPT:
            DosOpt(ciudad)
            #busqueda_local_trabajos2(ciudad,trabajos)
            #DosOpt_v2(ciudad,trabajos)
    
    value3 = random.uniform(0,1)
    if value3<MS2: 
        value = random.uniform(0, 1)
        if value < P_JLS:
            busqueda_local_trabajos2(ciudad,trabajos)
        else:
            perturbation(trabajos)

def busqueda_local_trabajos(ciudad,trabajos):
    n = len(ciudad)
    i_guardado = 0
    j_guardado = 0
    cmax, = costoTotal([ciudad,trabajos])
    flag = 0
    #print(trabajos)
    i = 0
    for i in range(len(ciudad)):
        if flag == 1:
            break
        for j in range(len(trabajos)):
            trabajos[i],trabajos[j]=trabajos[j],trabajos[i]
            nuevo_costo,= costoTotal([ciudad,trabajos])
            #print("ciudad ",ciudad)
            #print("trabajos ",trabajos)
            trabajos[j],trabajos[i]=trabajos[i],trabajos[j]
            if nuevo_costo<cmax:
                cmax = nuevo_costo
                i_guardado = i
                j_guardado = j
                flag = 1
                break
            #print(nuevo_costo)
    trabajos[i_guardado],trabajos[j_guardado]=trabajos[j_guardado],trabajos[i_guardado]

def busqueda_local_trabajos2(ciudad,trabajo):
    # print("ENTRÉ")
    # a=[[1, 2, 3, 4, 5], [5, 3, 4, 1, 2]]
    # ciudad = a[0]
    # trabajo = a[1]

    n = len(ciudad)
    suma_ac = [distancia(0, ciudad[0])]
    #print(distancia(0, ciudad[0]),0,ciudad[0])
    suma = suma_ac[-1]
    maxtime = suma_ac[-1]+jobTime(trabajo[0],ciudad[0]) 
    #cmax1 = [0]
    #i1_guardado = [0]
    cmax_antiguo = 0
    cmax2_antiguo = 0
    best_i = 0
    i2 = 0
    i = 0
    while i < n - 1:
        suma += distancia(ciudad[i], ciudad[i + 1])
        #print(distancia(ciudad[i], ciudad[i + 1]),ciudad[i], ciudad[i + 1])
        suma_ac.append(suma)
        maxtime = suma_ac[-1] + jobTime(trabajo[i+1],ciudad[i+1])  
        #print("costo2: ",suma_ac[-1],maxtime)
        if maxtime > cmax_antiguo:
            cmax2_antiguo = cmax_antiguo
            i2 = best_i
            cmax_antiguo = maxtime
            best_i = i+1
        elif maxtime > cmax2_antiguo:
            cmax2_antiguo = maxtime
            i2 = i

        i += 1
    suma += distancia(ciudad[-1], 0)
    suma_ac.append(suma)
    if suma_ac[-1]>cmax_antiguo:
        cmax2_antiguo = cmax_antiguo
        i2 = best_i
        best_i = n
        cmax1 = suma_ac[-1]
    
    #print("trabajo: ",trabajo)
    aux = -1
    for i in range(len(trabajo)):
        if i != best_i:
            cmax1_nuevo = suma_ac[best_i] + jobTime(trabajo[i],ciudad[best_i])
            cambio2 = suma_ac[i] + jobTime(trabajo[i],ciudad[i])
            #print("i: ",trabajo[i],trabajo[best_i])
            #print("nuevos: ",cmax1_nuevo,cambio2)
            if cmax1_nuevo < cmax_antiguo:
                if cmax1_nuevo > cambio2:
                    if cmax1_nuevo < cmax2_antiguo: 
                        aux = cmax2_antiguo
                        trabajo[best_i],trabajo[i] = trabajo[i],trabajo[best_i]
                        #print("entré 1")
                        break
                    else:
                        trabajo[best_i],trabajo[i] = trabajo[i],trabajo[best_i]
                        #print("entré 2")
                        aux = cmax1_nuevo
                        break
                else: #cmax nuevo mayor o igual al segundo mas grande 
                    if cambio2>cmax2_antiguo:
                        trabajo[best_i],trabajo[i] = trabajo[i],trabajo[best_i]
                        #print("entré 3")
                        aux = cambio2
                        break
                    else:
                        aux = cmax2_antiguo
                        #print("entré 4")
                        trabajo[best_i],trabajo[i] = trabajo[i],trabajo[best_i]
                        break
    """
    #print(cmaxs)
    #print("entré2 ",ciudad,trabajo)
    # aux3, = costoTotal([ciudad,trabajo])
    # if aux3 != aux and aux != -1:
    #     print("cmax: ",aux3,aux)
    #     print("ERROR")
    #     exit(0)
    """

def GA(ciudad,comparar,plot):
    n = len(ciudad)
    global N
    N = n
    # poblacion = 50
    elit_size = POBLACION*ELITE

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    #creator.create("Individual", numpy.ndarray, typecode='i', fitness=creator.FitnessMin)
    creator.create("Individual", list, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator
    #toolbox.register("indices", random.sample, range(n), n)
    #depot = random.randrange(0, n)
    #toolbox.register("indices", vecinoMasCercano, n)
    t_aux = time.time()
    toolbox.register("cromo1", generarRuta, n-1)
    toolbox.register("cromo2", creador_trabajos, n)
    toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.cromo1, toolbox.cromo2), n=1)

    # Structure initializers
    #toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    tiempo_poblacion = time.time()-t_aux

    toolbox.register("mate",cruzamiento)
    #toolbox.register("mate",PMX)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selTournament, tournsize=TOURN)
    toolbox.register("evaluate", costoTotal)
    pop = toolbox.population(n=POBLACION)

    #hof = tools.HallOfFame(elit_size, similar=numpy.array_equal)
    hof = tools.HallOfFame(elit_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    log = tools.Logbook()
    log.header = "gen", "evals", "std", "min", "avg", "max"
    #CXPB, MUTPB = 0.9, 0.2
    inicioTiempo = time.time()
    t_aux = time.time()
    for i in pop:
        i[1] = crear_trabajos2(i[0],i[1])
    tiempo_poblacion += time.time()-t_aux 
    # Población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    promedio_p_inicial = numpy.mean(fitnesses)
    mejor_inicial = numpy.min(fitnesses)
    mejor_aux = mejor_inicial
    # Evaluación
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    iterMax = IT
    record = stats.compile(pop)
    log.record(gen=g, evals=len(pop), **record)
    hof.update(pop)
    #hof_size = len(halloffame.items) if halloffame.items else 0
    
    if comparar == False:
        print(log[-1]["gen"], "%.2f"%log[-1]["avg"], "%.2f"%log[-1]["min"])
    #lista_soluciones = []
    #lista_costos = []

    if comparar == True:
        df = pd.DataFrame(columns=["gen","avg","min","std"])
        df.loc[0] = [log[-1]["gen"], "%.2f"%log[-1]["avg"], "%.2f"%log[-1]["min"], "%.2f"%log[-1]["std"]]
    timeLimit = 1800
    iteracion_mejor = 0
    # Proceso evolutivo
    while g < iterMax and time.time() - inicioTiempo <= timeLimit:
        g = g + 1

        # Selección
        offspring = toolbox.select(pop, int(len(pop)-elit_size))

        # Cruzamiento
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                toolbox.mate(child1[1], child2[1])
                del child1.fitness.values
                del child2.fitness.values

        #Mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0],mutant[1])
                del mutant.fitness.values               

        # Evaluación
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if fit < mejor_aux:
                mejor_aux = fit 
                iteracion_mejor = g

        #best = hof.items[0]
        #DosOpt(hof.items[0][0])
        # aux = hof.items[0][1]
        # if aux != hof.items[0][1]:
        busqueda_local_trabajos2(hof.items[0][0],hof.items[0][1])
        #     print("distintos")
        

        
        #ELITISMO
        offspring.extend(hof.items)
        hof.update(offspring)

        # Reemplazo
        pop[:] = offspring
        #fits = [ind.fitness.values[0] for ind in pop]
        #print( "fitness-- ", max(fits))
        record = stats.compile(offspring)
        log.record(gen=g, evals=len(offspring), **record)
        if comparar == False:
            print(log[-1]["gen"], "%.2f"%log[-1]["avg"], "%.2f"%log[-1]["min"])

        top = tools.selBest(offspring, k=1)
        # print(top[0]) 
        #s = top[0]
        #costo = costoTotal(top[0])
        #lista_costos.append(int(log[-1]["min"]))
        #lista_soluciones.append(top[0][0:n])
        if comparar == True:
            df.loc[g] = [log[-1]["gen"], "%.2f"%log[-1]["avg"], "%.2f"%log[-1]["min"],"%.2f"%log[-1]["std"]]

    #print(lista_soluciones[-1])
    #print(top[-1])
    finTiempo = time.time()
    tiempo = finTiempo - inicioTiempo
    minimo, promedio = log.select("min", "avg")
    if comparar == False:
        print('Costo  : %.2f' % log[-1]["min"])
        print("Tiempo : %f" % tiempo)
        if plot == True:
            plots = plt.plot(minimo,'c-', promedio, 'b-')
            #print( log.select('mean'))
            plt.legend(plots, ('Costo Mínimo', 'Costo Promedio'), frameon=True)
            plt.ylabel('Costo')
            plt.xlabel('Iteraciones')
            plt.show()

    else:
        return "%.2f"%log[-1]["min"],tiempo,"%.2f"%log[-1]["avg"],df,iteracion_mejor,promedio_p_inicial,mejor_inicial,tiempo_poblacion

tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
semilla = 1
batch = 1
instancia = "gr17"
size = "tsplib" if instancia in tsplib else "Small"

#Cruzamiento
P_OX      = 0.5
P_PMX     = 0.25
P_UMPX    = 0.25

#Tour
P_NNH     = 0.5
P_TSP     = 0.4
P_RPT     = 0.1

#job
P_NNHJ    = 0.7
P_RPJ     = 0.3

#Mutacion
MS1       = 0.5
MS2       = 0.5
P_EM      = 0.25 
P_RM      = 0.25 
P_SM      = 0.25 
P_2OPT    = 0.25 
P_JLS     = 0.3
P_JEM     = 0.7 #Exchange mutation job

#Overall
POBLACION = 50
CXPB      = 0.9
MUTPB     = 0.2
IT        = 500
ELITE     = 0.1
TOURN     = 4

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]
multi = False
mbool = True

for i in range(len(opts)):
    if opts[i][0][1:] == "multi":  multi  = (opts[i][1])
    elif   opts[i][0][1:] == "seed": semilla = int(opts[i][1])
    #elif opts[i][0][1:] == "tipo": tipo = opts[i][1] 
    elif opts[i][0][1:] == "size": size = str(opts[i][1]) 
    elif opts[i][0][1:] == "batch" : batch = int(opts[i][1]) 
    elif opts[i][0][1:] == "i":
        try:
            instancia = int(opts[i][1])  
        except ValueError:
            instancia = opts[i][1]      
    elif opts[i][0][1:] == "OX"   : P_OX    =  float(opts[i][1])
    elif opts[i][0][1:] == "PMX"  : P_PMX   =  float(opts[i][1])  
    elif opts[i][0][1:] == "UMPX" : P_UMPX  =  float(opts[i][1])  

    elif opts[i][0][1:] == "NNH"  : P_NNH   =  float(opts[i][1]) 
    elif opts[i][0][1:] == "TSP"  : P_TSP   =  float(opts[i][1]) 
    elif opts[i][0][1:] == "RPT"  : P_RPT   =  float(opts[i][1]) 
    elif opts[i][0][1:] == "NNHJ" : P_NNHJ  =  float(opts[i][1]) 
    elif opts[i][0][1:] == "RPJ"  : P_RPJ   =  float(opts[i][1]) 

    elif opts[i][0][1:] == "MS1"  : MS1     =  float(opts[i][1])
    elif opts[i][0][1:] == "MS2"  : MS2     =  float(opts[i][1]) 
    elif opts[i][0][1:] == "EM"   : P_EM    =  float(opts[i][1])  
    elif opts[i][0][1:] == "RM"   : P_RM    =  float(opts[i][1])  
    elif opts[i][0][1:] == "SM"   : P_SM    =  float(opts[i][1]) 
    elif opts[i][0][1:] == "OPT2" : P_2OPT  =  float(opts[i][1])   
    elif opts[i][0][1:] == "JLS"  : P_JLS   =  float(opts[i][1])  
    elif opts[i][0][1:] == "JEM"  : P_JEM   =  float(opts[i][1])  

    elif opts[i][0][1:] == "POB": POBLACION = int(opts[i][1]) 
    elif opts[i][0][1:] == "CXPB" : CXPB   = float(opts[i][1]) 
    elif opts[i][0][1:] == "MUTPB": MUTPB  = float(opts[i][1])
    elif opts[i][0][1:] == "IT"   : IT     = int(opts[i][1])
    elif opts[i][0][1:] == "ELITE" : ELITE  = float(opts[i][1])
    elif opts[i][0][1:] == "TOURN": TOURN  = int(opts[i][1])

if isinstance(instancia, int):
    #batch = (instancia-1)//25+1
    if instancia <=25: batch = 1
    elif instancia <=50: batch = 2
    elif instancia <=75: batch = 3
    else: batch = 4

if multi ==False:
    print("Instancia :",instancia)
    print("SEED: ",semilla)

if instancia not in tsplib: #Si es que es instancia de prueba
    random.seed(semilla)
    ciudad,arch,TT,JT = parameters(size,batch,instancia)
    m,t,p,d,mejor_iteracion,promedio_inicial,mejor_inicial,t_poblacion = GA(ciudad,mbool,False)
    if multi==False:
        print("tiempo: ",t," segundos")
        print("Mejor","%.2f"%float(m))
    else:
        #semilla,size,batch,instancia,mejor,poblacion,tiempo,mejor inicial,poblacion inicial,mejor iteracion,tiempo_poblacion
        print("{:<6}{:<8}{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format(semilla,size,batch,instancia,"%.2f"%float(m),"%.2f"%float(p),"%.3f"%t,"%.2f"%float(mejor_inicial),"%.2f"%float(promedio_inicial),mejor_iteracion,round(t_poblacion,2)))

else:
    random.seed(semilla)
    ciudad,arch,TT,JT = parametersexcel(instancia)
    m,t,p,d,mejor_iteracion,promedio_inicial,mejor_inicial,t_poblacion = GA(ciudad,mbool,False)
    if multi==False:
        print("tiempo: ",t," segundos")
        print("Mejor","%.2f"%float(m))
    else:
        #semilla,instancia,mejor,poblacion,tiempo,mejor inicial, poblacion inicial, iteracion mejor,tiempo_poblacion
        print("{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format(semilla,instancia,"%.2f"%float(m),"%.2f"%float(p),"%.3f"%t,"%.2f"%float(mejor_inicial),"%.2f"%float(promedio_inicial),mejor_iteracion,round(t_poblacion,2)))

#[[15, 11, 8, 3, 12, 6, 7, 5, 16, 13, 14, 2, 10, 4, 9, 1], [3, 8, 11, 16, 2, 13, 12, 14, 1, 6, 5, 15, 9, 10, 4, 7]]