from pyexpat import model
import gurobipy as gp
from gurobipy import *
import pandas as pd
from itertools import combinations
import numpy as np
import os
import tsplib95
import lkh

path = "Codigos/"
#path=""

def transformar_txt(size,batch,instancia,ruta,n):
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
    
    archivo = open(path+size+"_"+str(batch)+"_"+str(instancia)+".txt","w")
    archivo.write("NAME: prueba"+str(n+1)+"\n")
    archivo.write("TYPE: TSP\n")
    archivo.write(f"COMMENT: {n+1} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n")
    archivo.write(f"DIMENSION: {n+1}\n")
    archivo.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n")
    for i in lineas:
        archivo.write(i[0]+"\n")
    archivo.close()

def solve_lkh(size,batch,instancia,n):
    ruta_instancia = path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_cost_table_by_coordinates.csv"
    transformar_txt(size,batch,instancia,ruta_instancia,n)
    #problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
    problem = tsplib95.load(path+size+"_"+str(batch)+"_"+str(instancia)+".txt")
    solver_path = path+'LKH-3.0.7/LKH'
    ciudad = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)[0]
    ciudad = [i-1 for i in ciudad]
    os.remove(path+size+"_"+str(batch)+"_"+str(instancia)+".txt")
    return ciudad

def heuristica_trabajos(ruta,JT):
    
    ciudad = ruta
    trabajo = []
    n = len(ciudad)  
    cont = len(ruta)-1
    #QUE RECORRA HACIA ATRÁS PARTIENDO DE DISTINTOS NODOS
    while len(trabajo)<n:
        tiempos = {(i,ciudad[cont]):JT[(i,ciudad[cont])] for i in range(1,n+1)}
        nuevo = min(tiempos.items(),key=lambda x:x[1]) 
        trabajo.append(nuevo[0][1])
        cont -= 1
    return trabajo

def ordenar_arcos(ruta):
    arcos = []
    for i in range(len(ruta)-1):
        arcos.append((str(ruta[i]),str(ruta[i+1])))
    arcos.append((str(ruta[-1]),str(ruta[0])))
    return arcos

def ordenar_trabajos(ruta,trabajos):
    return [(ruta[i],trabajos[i]) for i in range(len(ruta))]


def distancia(i, j):
    global TT
    return TT[(i,j)]

def jobTime(i,j):
    global JT
    #JT[(trabajos,nodos)]
    return JT[(i,j)]

def parametersexcel(excel):
    ruta = path+"Data/instancias_paper/"+excel+".xlsx"
    TT = pd.read_excel(ruta,sheet_name="TT",index_col=0)
    JT = pd.read_excel(ruta,sheet_name="JT",index_col=0)
    #coord = pd.read_excel(excel,sheet_name = "coord",index_col=0,header=0)
    #coord_x = coord["coord_x"]
    #coord_y = coord["coord_y"]
    #print(JT[2][1])
    #JT[NODOS][TRABAJOS]
    n = len(list(JT.loc[0].dropna()))
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
            modelo.cbLazy(gp.quicksum(modelo._vars[i, j] for i, j in combinations(tour, 2)) <= len(tour)-1)

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

    TT = pd.read_csv(f"{path}Data/test/TT_paper.csv",index_col= None, header = None)
    JT = pd.read_csv(f"{path}Data/test/JT_paper.csv",index_col= None, header = None)

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}
    return nodes,arch,travel_time,job_time

def gurobi(tipo_instancia,subtour,sumarM,sol_inicial,output):
    """
    subour:\n
          0=Sin restricciones 14-15 de subtour\n
          1=Restricciones GG\n
          2=Restricciones MTZ\n 
          3=Restricciones DL
    """
    if subtour==0:
        print("Instancias: ",tipo_instancia,", Sin subtour")
    elif subtour == 1:
        print("Instancias: ",tipo_instancia,"GG")
    elif subtour == 2:
        print("Instancias: ",tipo_instancia,"MTZ")
    elif subtour == 3:
        print("Instancias: ",tipo_instancia,"DL")

    if tipo_instancia=="tsplib":
        instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
    else:
        instancias = [u for u in range(1,101)]
        batch = [u//25+1 for u in range(100)]


    #for v in range(len(instancias)):
    for v in range(1):
        global TT
        global JT
        if tipo_instancia=="tsplib":
            ciudades,arcos,TT,JT = parametersexcel(instancias[v])
        
        elif tipo_instancia.lower() == "small":
            ciudades,arcos,TT,JT = parameters("Small",batch[v],instancias[v])
            tsp_inicial = solve_lkh("Small",1,1,len(ciudades)-1)

        elif tipo_instancia.lower() == "medium":
            ciudades,arcos,TT,JT = parameters("Medium",batch[v],instancias[v])
            tsp_inicial = solve_lkh("Medium",1,1,len(ciudades)-1)

        elif tipo_instancia.lower() == "large":
            ciudades,arcos,TT,JT = parameters("Large",batch[v],instancias[v])
            tsp_inicial = solve_lkh("Large",1,1,len(ciudades)-1)
        
        else:
            print("instancia erronea")
            exit(0)


        M = 0
        for i in range(len(ciudades)):
            maximo_t = 0
            maximo_tt = 0
            for j in range(len(ciudades)):
                if i!=j and TT[(i,j)]>maximo_t:
                    maximo_t = TT[(i,j)]
                
                if j!= 0 and JT[(j,i)]>maximo_tt:
                    maximo_tt = JT[(j,i)]
            M += maximo_t #+maximo_tt      
        M += sumarM
        #print("M",M)

        dist = {(i, j):distancia(i,j) for i, j in arcos}
        trabajos = ciudades.copy()
        nodos_trabajos = [(i,k) for i in ciudades for k in ciudades]
        jt_aux = np.array(list(JT.values()))
        jt_min = jt_aux[(jt_aux >= 1)].min()
        



        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',output)
            env.start()
            with gp.Model(env=env) as modelo:
                Cmax = modelo.addVar(name="Cmax")

                x = modelo.addVars(dist.keys(), vtype=GRB.BINARY, name='x')
                z = modelo.addVars(nodos_trabajos, vtype=GRB.BINARY, name='z')

                if subtour==1: #Variable GG
                    y = modelo.addVars(arcos,name = "y") 
                    
                elif subtour==2: #Variable MTZ y DL
                    u = modelo.addVars(ciudades , vtype = GRB.CONTINUOUS , name = "u")

                elif subtour == 3:
                    u = modelo.addVars(ciudades , vtype = GRB.CONTINUOUS , name = "u")

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

                #Restricción GG
                if subtour == 1:
                    for i in ciudades[1:len(ciudades)]: #14
                        modelo.addConstr(quicksum(y[(i,j)] for j in ciudades if i !=j) - quicksum(y[(j,i)] for j in ciudades if i !=j) == 1)

                    for i in ciudades[1:len(ciudades)]: #15
                        for j in ciudades:
                            if i!=j:
                                modelo.addConstr(y[(i,j)]<= len(ciudades)*x[(i,j)])
                
                # #Restricción MTZ
                elif subtour ==2:
                    for i,j in arcos:
                        if i>0:
                            modelo.addConstr(u[i] - u[j] + 1 <= M * (1 - x[(i,j)]) , "MTZ(%s,%s)" %(i, j))
                        
                #Restricción DL proposition 3
                elif subtour == 3:
                    n = len(ciudades)
                    for i in range(1,len(ciudades)):
                        modelo.addConstr(u[i] >= 1 + (n-3)*x[(i,0)] + quicksum(x[(j,i)] for j in ciudades[1:] if j != i))

                    for i in range(1,len(ciudades)):
                        modelo.addConstr(u[i] <= n-1 - (n-3)*x[(0,i)]- quicksum(x[(i,j)] for j in ciudades[1:] if j != i))

                for i in ciudades: #16
                    for j in ciudades[1:len(ciudades)]:
                        if i!=j:
                            modelo.addConstr(TS[i] + TT[(j,i)] - (1-x[(i,j)])*M <= TS[j])


                
                # Restricción 2-degree
                #modelo.addConstrs(x.sum(i, '*') == 2 for i in range(len(ciudades)))

                # Parámetros
                #modelo.Params.LazyConstraints = 1
                modelo.Params.Threads = 1
                modelo.setParam('TimeLimit', 3600)
                if sol_inicial == True and tipo_instancia != "tsplib":
                    modelo.NumStart = 1
                    modelo.update()

                    # imprimir modelo

                    # Solucion inicial
                    if tipo_instancia != "tsplib":
                        arcos_inicial = ordenar_arcos(tsp_inicial)
                        trabajos = heuristica_trabajos(tsp_inicial[1:],JT)
                        arcos_inicial_trabjos = ordenar_trabajos(tsp_inicial[1:],trabajos)
                        for s in range(modelo.NumStart):
                            # set StartNumber
                            modelo.Params.StartNumber = s
                            # now set MIP start values using the Start attribute, e.g.:
                            for var in modelo.getVars(): #
                                if var.VarName[0] == "x":
                                    nombre_arco = tuple(var.varName.split("[")[1][:-1].split(","))
                                    if nombre_arco in arcos_inicial:
                                        var.Start = 1
                                    # else:
                                    #     var.Start = 0

                                elif var.VarName[0] == "z":
                                    nombre_arco = tuple(var.varName.split("[")[1][:-1].split(","))
                                    if nombre_arco in arcos_inicial_trabjos:
                                        var.Start = 1
                                    # else:
                                    #     var.Start = 0
                                
                                elif var.VarName == "Cmax":
                                    #var.Start = <valor>
                                    pass

                


                modelo.optimize()
                
                #modelo.write("file2.lp")

                lower = modelo.ObjBoundC
                objective = modelo.getObjective().getValue()

                gap = round((objective-lower)/lower*100,4)

                lower = round(modelo.ObjBoundC,4)
                objective = round(modelo.getObjective().getValue(),4)
                time = round(modelo.Runtime,2)
                # instancia, bks, lower, gap, time
                print("{:<10}{:<10}{:<10}{:<10}{:<10}".format(instancias[v],objective,lower,gap,time))

#tsplib, Small, Medium, Large
tipo = "Large"

gurobi(tipo , subtour = 0 , sumarM = 0 , sol_inicial = True , output = 1)
# gurobi(tipo , subtour = 1 , sumarM = 0 , sol_inicial = True , output = 0)
# gurobi(tipo , subtour = 2 , sumarM = 0 , sol_inicial = True , output = 0)
# gurobi(tipo , subtour = 3 , sumarM = 0 , sol_inicial = True , output = 0)

