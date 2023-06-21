import gurobipy as gp
from gurobipy import *
import pandas as pd
from itertools import combinations
import numpy as np
import os
import tsplib95
import lkh
import sys
import subprocess


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

def costoTotal(ciudad2):

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


    return cmax,


def solve_lkh(size,batch,instancia,n):
    if output == True:
        print("Creando archivo .txt ...")
    
    if size != "tsplib":
        ruta_instancia = path+"Data/"+str(size)+"_problems/Batch_0"+str(batch)+"/TSPJ_"+str(instancia)+size[0]+"_cost_table_by_coordinates.csv"
    else:
        ruta_instancia = path+f"Data/Tsplib_problems/TT_{instancia}.csv"

    transformar_txt(size,batch,instancia,ruta_instancia,n)
    
    problem = tsplib95.load(path+size+"_"+str(batch)+"_"+str(instancia)+".txt")
    os.remove(path+size+"_"+str(batch)+"_"+str(instancia)+".txt")

    solver_path = path+'LKH-3.0.7/LKH'

    ciudad = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)[0]
    ciudad = [i-1 for i in ciudad]
    if output==True:
        print("Archivo .txt eliminado")
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
    if donde == GRB.Callback.MIPNODE:
        #valoresX = modelo.cbGetSolution(modelo._vars)
        valoresX = model.cbGetNodeRel(model._vars)
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

    TT = pd.read_csv(f"{path}Data/test/1_TT_paper.csv",index_col= None, header = None)
    JT = pd.read_csv(f"{path}Data/test/1_JT_paper.csv",index_col= None, header = None)

    #print(JT[2][1])
    #JT[COLUMNA][FILA]
    #TT[COLUMNA][FILA]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}
    return nodes,arch,travel_time,job_time

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

def obtener_cortes():
    archivo = open(f"{path}logfile_{instancia}.txt","r")
    lineas = [linea.split("\n") for linea in archivo]
    archivo.close()
    #os.remove(f"{path}logfile_{instancia}.txt")
    cortes = {}
    aux = []
    flag = 0
    contador = 0
    for linea in lineas[::-1]:
        if "Explored" in linea[0]:
            flag = 1
            continue
        
        if flag == 1:
            aux.append(linea)
            contador +=1
        
        if contador > 25:
            return 0

        if "Cutting planes:" in linea:
            break
    
    del linea
    aux.pop(0)
    aux.pop(-1)
    total = 0
    for i in range(len(aux)):
        nombre = aux[i][0].split(":")[0][2:]
        cantidad = int(aux[i][0].split(":")[1])
        total += cantidad
        cortes[nombre] = cantidad
    del aux
    return total

def subtour2(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys()
                         if vals[i, j] > 0.5)
    unvisited = list(range(n_ciudades))
    cycle = range(n_ciudades+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

def subtourelim2(model, where):
    # global contador
    # global restricciones
    if where == GRB.Callback.MIPNODE:
        vals = model.cbGetNodeRel(model._vars)
        #vals = model.cbGetSolution(model._vars)
        
        # find the shortest cycle in the selected edge list
        tour = subtour2(vals)
        #rutas.append(tour)
        #arcos.append([tupla for tupla, valor in vals.items() if valor >0.5])
        #print(len(tour),n_ciudades,[valor for tupla, valor in vals.items() if valor >0])
        #print("Intenté")
        #vals_z = model.cbGetSolution(model._varsz)
        # aux1 = [(i,j) for i,j in vals.items() if j >0]
        # aux2 = [(i,j) for i,j in vals_z.items() if j >0]
        # print(aux1)
        # print(aux2)
        if len(tour) < n_ciudades:
            #print("ENTRE1")
            model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))<= len(tour)-1)
            #print("+".join([f"({i},{j})" for i,j in combinations(tour, 2)])+f"<= {len(tour)-1}")
        #restricciones.append(contador)

contador_callback = 0
def subtourelim3(modelo, donde):
    global contador_callback
    n = n_ciudades
    if donde == GRB.Callback.MIPNODE:
        #valoresX = modelo.cbGetSolution(modelo._vars)
        #depth = modelo.cbGet(GRB.Callback.MIPNODE_NODCNT)
        valoresX = modelo.cbGetNodeRel(modelo._vars)
        # encontrar ciclo más pequeño
        tour = [i for i in range(n+1)]
        subtour3(tour, valoresX,n)
        if len(tour) < n:
            contador_callback +=1 
            # agregar cortes de elimination de subtour DFJ2
            tour2 = [i for i in range(n) if i not in tour]
            modelo.cbLazy(gp.quicksum(modelo._vars[i, j] for i in tour for j in tour2) >= 1)
            #print(depth)

        valoresZ = modelo.cbGetNodeRel(modelo._varsz)
        solucion = [(arco,solucion) for arco,solucion in valoresZ.items() if solucion >0 and solucion <1]
        if len(solucion) >0:
            solucion = [(arco,solucion) for arco,solucion in valoresZ.items() if solucion >0 ]
            for i in solucion: print(i)
            exit(0)

            

def subtour3(subruta, vals,n):
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

def gurobi_solver(tipo_instancia,instancia,subtour,sol_inicial,output,sumarM=0):
    """
    Tipo: tsplib/Small/Medium/Large\n
    subtour: 0=Sin restricciones 14-15 de subtour\n
             1=Restricciones GG\n
             2=Restricciones MTZ\n 
             3=Restricciones DL\n
    sumarM: Cantidad a agregar al M de las restricciones\n
    sol_inicial: True/False, para activar solucion inicial\n
    output: True/False, para mostrar output de gurobi
    """
    # if subtour==0:
    #     print("Instancias: ",tipo_instancia,", Sin subtour")
    # elif subtour == 1:
    #     print("Instancias: ",tipo_instancia,"GG")
    # elif subtour == 2:
    #     print("Instancias: ",tipo_instancia,"MTZ")
    # elif subtour == 3:
    #     print("Instancias: ",tipo_instancia,"DL")

    if tipo_instancia != "tsplib":
        if instancia<=25: batch = 1
        elif instancia<=50: batch = 2
        elif instancia<=75: batch = 3
        else: batch = 4



    global TT
    global JT
    if tipo_instancia.lower()=="tsplib":
        ciudades,arcos,TT,JT = parametersexcel(instancia)
        try:
            tsp_inicial = solve_lkh("tsplib","",instancia,len(ciudades)-1)
        except:

            print(instancia)
            exit(0)
    elif tipo_instancia.lower() == "small":
        ciudades,arcos,TT,JT = parameters("Small",batch,instancia)
        tsp_inicial = solve_lkh("Small",batch,instancia,len(ciudades)-1)

    elif tipo_instancia.lower() == "medium":
        ciudades,arcos,TT,JT = parameters("Medium",batch,instancia)
        tsp_inicial = solve_lkh("Medium",batch,instancia,len(ciudades)-1)

    elif tipo_instancia.lower() == "large":
        ciudades,arcos,TT,JT = parameters("Large",batch,instancia)
        tsp_inicial = solve_lkh("Large",batch,instancia,len(ciudades)-1)
    
    else:
        
        #ciudades,arcos,TT,JT = parameterscsv()
        print("instancia erronea")
        exit(0)


    arcos_inicial = ordenar_arcos(tsp_inicial)
    trabajos = heuristica_trabajos(tsp_inicial[1:],JT)
    arcos_inicial_trabjos = ordenar_trabajos(tsp_inicial[1:],trabajos)

    costo_inicial = costoTotal([tsp_inicial[1:],trabajos])[0]
    menor_arco_depot = min(TT[(0,i)] for i in range(1,len(ciudades)))


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

    #####Calculo de nueva M
    M2 = np.zeros((len(ciudades),len(ciudades)))
    for j in range(1,len(ciudades)):
        M2[0,j] = TT[(0,j)]
    for i in range(len(ciudades)):
        for j in range(len(ciudades)):
            if i != j:
                M2[i,j] = costo_inicial+TT[(i,j)]
    



    dist = {(i, j):distancia(i,j) for i, j in arcos}
    trabajos = ciudades.copy()
    nodos_trabajos = [(i,k) for i in ciudades for k in ciudades]
    jt_aux = np.array(list(JT.values()))
    jt_min = jt_aux[(jt_aux >= 1)].min()
    
    global n_ciudades
    n_ciudades = len(ciudades)



    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',1 if output else 0)#1 if output else 0)
        #env.setParam('LogFile',f"{path}logfile_{instancia}.txt")
        
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

            #modelo.addConstr(Cmax <= 3500)

            # como es simétrico copiamos sus opuestos
            # for i, j in x.keys():
            #     x[j, i] = x[i, j]

            # Restricción 2-degree
            #modelo.addConstrs(x.sum(i, '*') == 2 for i in range(len(ciudades)))

            # for i in range(len(ciudades)):
            #     modelo.addConstr(sum(x[i,j] for j in range(len(ciudades)) if i != j) == 1)    

            modelo.addConstrs(x.sum(i,'*') == 1 for i in ciudades) # Outgoing
            modelo.addConstrs(x.sum('*', j) == 1 for j in ciudades) # Incoming 

            #Restricciones
            for i in ciudades[1:len(ciudades)]:
                modelo.addConstr(Cmax >= TS[i] + quicksum(z[(i,k)]*JT[(k,i)] for k in trabajos if k!=0 ))

            for i in ciudades[1:len(ciudades)]:
                modelo.addConstr(Cmax >= TS[i] + x[(i,0)]*TT[(0,i)])

            for k in trabajos[1:len(ciudades)]:
                modelo.addConstr(quicksum(z[(i,k)] for i in ciudades if i != 0) == 1)

            for i in ciudades[1:len(ciudades)]: 
                modelo.addConstr(quicksum(z[(i,k)] for k in trabajos if k != 0) == 1)
            
            # for i in ciudades: #12
            #     modelo.addConstr(quicksum(x[(i,j)] for j in ciudades if i!=j)==1)

            # for j in ciudades: #13
            #     modelo.addConstr(quicksum(x[(i,j)] for i in ciudades if i!=j)==1) 



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

            #Cota superior tiempo de inicio
            for i in ciudades:
                modelo.addConstr(TS[i]<=costo_inicial)

            #Cota inferior tiempo de inicio
            for i in range(1,len(ciudades)):
                modelo.addConstr(TS[i]>=menor_arco_depot) 

            #Cota superior de solucion inicial (lkh+NNJ)
            modelo.addConstr(Cmax<=costo_inicial)


            # Parámetros
            modelo.Params.LazyConstraints = 1
            modelo.Params.Threads = 1
            modelo.setParam('TimeLimit', 1500)
            #modelo.setParam("NodeLimit",1)
            
            if sol_inicial == True: #and tipo_instancia != "tsplib":
                modelo.NumStart = 1
                modelo.update()

                # imprimir modelo

                # Solucion inicial

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

            modelo._vars = x
            modelo._varsz = z
            #modelo.optimize()
            modelo.optimize(subtourelim3)
            dict_status = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}
            #modelo.write("file2.lp")

            lower = modelo.ObjBoundC
            gap = float("inf")
            objective = float("inf")
            if modelo.Status == GRB.OPTIMAL or modelo.SolCount > 0:
                objective = modelo.getObjective().getValue()
                lower = modelo.ObjBoundC
                #print("Status", modelo.Status, GRB.OPTIMAL)
                gap = round((objective-lower)/lower*100,4)
                lower = round(lower,4)
                objective = round(objective,4)

            # else:
            #     #print(instancia, ':Optimization ended with status %d' % modelo.Status)
            #     if modelo.SolCount > 0:
            #         objective = modelo.getObjective().getValue()
            #         lower = modelo.getObjective().getValue()
            #         #print("Status", modelo.Status, GRB.OPTIMAL)
            #         gap = round((objective-lower)/lower*100,4)
            #         lower = round(lower,4)
            #         objective = round(objective,4)

            if print_solucion:
                solucion = []
                for i in x:
                    if x[i].X>0:
                        solucion.append(i)
                solucion = ordenar_solucion(solucion,len(ciudades))[1:-1]

                solucion_trabajos = {}
                for i in z:
                    if z[i].X>0:
                        #print(i[0],":",i[1])
                        solucion_trabajos[i[0]]=i[1]
                #print(solucion_trabajos)

                trabajos = []
                for i in solucion:
                    trabajos.append(solucion_trabajos[i])

                print(f"[{solucion},{trabajos}]")

            time = round(modelo.Runtime,4)
            # instancia, bks, lower, gap, time,nodos explorados
            # obtener lower bound en root node o nodo 0
            
            print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}".format(instancia,round(objective,2),round(lower,1),round(gap,4),time,dict_status[modelo.Status],modelo.SolCount,modelo.NodeCount),end=" ")
            #print(modelo.getParamInfo("ZeroHalfCuts"))
            print(contador_callback)#,"restricciones")
            #print(obtener_cortes())
            

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]
tipo      = "tsplib"
instancia = "gr17"
t_sub     = 0
sol_in    = True
output    = True
sumar_m   = 0
print_solucion = False


for i in range(len(opts)):
    if opts[i][0][1:] == "tipo":  tipo  = (opts[i][1])
    elif   opts[i][0][1:] == "subtour": t_sub = int(opts[i][1])
    elif opts[i][0][1:] == "instancia":
        try:
            instancia = int(opts[i][1])  
        except ValueError:
            instancia = opts[i][1]  
    elif   opts[i][0][1:] == "solinicial": sol_in = True if str(opts[i][1]) =="True" else False
    elif   opts[i][0][1:] == "output": output = True if str(opts[i][1]) =="True" else False
    elif   opts[i][0][1:] == "sumarM": sumar_m = int(opts[i][1])

#print(tipo,instancia,t_sub,sol_in,output,sumar_m)
gurobi_solver(tipo,instancia,subtour=t_sub,sol_inicial=sol_in,output=output,sumarM=sumar_m)
