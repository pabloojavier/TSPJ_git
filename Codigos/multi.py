from ast import arg
import itertools
import os
import multiprocessing
import time
import random
import sys


#/usr/local/bin/python3.9 multi.py -p paralelo -t tsplib
path = "Codigos/"

inicio = time.time()
def launcher1(semilla,paralelo,_instancia,**kwargs):
    ejecutar = f"/usr/local/bin/python3 {path}ga_05.py -paralelo {paralelo} -seed {semilla} -i {_instancia}"
    for key,value in kwargs.items():
        if key != '-p':
            ejecutar += f"-{key} {value} "
    os.system(ejecutar)

def launcher2(semilla,paralelo,_size,_batch,_instancia ,**kwargs):
    ejecutar = f"/usr/local/bin/python3 {path}ga_05.py -paralelo {paralelo} -seed {semilla} -size {_size} -i {_instancia}"
    for key,value in kwargs.items():
        if key != '-p':
            ejecutar += f"-{key} {value} "
    os.system(ejecutar)

def launcher3(tipo,instancia,subtour,solin,output,sumarm):
    os.system(f"/usr/local/bin/python3 {path}MM_gurobi.py -tipo {tipo} -instancia {instancia} -subtour {subtour} -solinicial {solin} -output {output} -sumarM {sumarm}")

#Valores por defecto
paralelo = "paralelo"
size = "small"
batch = 1
instancia = "1"
alg = "ag"
subtour = None

P_RPT     = 0.236 #0.1
P_NNH     = 0.112 #0.5 
P_TSP     = 0.652 #0.4

#job        
P_RPJ     = 0.497 #0.3
P_NNHJ    = 0.503 #0.7

#Cruzamiento
P_OX      = 0.429400386847195 #0.5 #12
P_PMX     = 0.362669245647969 #0.25 #20
P_UMPX    = 0.207930367504836  #0.25 #68

#Mutacion   
MS1       = 0.121  #0.5
MS2       = 0.883  #0.5
P_EM      = 0.338612779901693  #0.25 
P_RM      = 0.252867285636264 #0.25 
P_SM      = 0.395412342981977 #0.25 
P_2OPT    = 0.0131075914800655 #0.25 
P_JLS     = 0.914 #0.3
P_JEM     = 0.086 #0.7 #Exchange mutation job, Complemento de JLS

#Overall    
ELITE     = 0.147 #0.1
POBLACION = 100 #50
CXPB      = 0.327  #0.9
MUTPB     = 0.717  #0.2
IT        = 500 #500
TOURN     = 2 #4


tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
#argv = sys.argv[1:]

argv = ["-p","paralelo","-size","tsplib","-alg","ag","-subtour","1"]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]
if len(opts)<3:
    print("multi.py -p <paralelo/secuencial> -size <tsplib/Small/Medium/Prueba> -alg <ag/gurobi>")
    exit(0)
if len(argv)%2 != 0:
    print("Error, falta un parametro o valor")
    exit(0)

for i in range(len(opts)):
    if opts[i][0][1:] == "p": paralelo = opts[i][1]
    elif opts[i][0][1:] == "size": size = str(opts[i][1])
    elif opts[i][0][1:] == "alg": alg = str(opts[i][1])
    elif opts[i][0][1:] == "subtour": subtour = int(opts[i][1])
    elif opts[i][0][1:] == "OX"   : P_OX    =  float(opts[i][1])
    elif opts[i][0][1:] == "PMX"  : P_PMX   =  float(opts[i][1])  
    elif opts[i][0][1:] == "UMPX" : P_UPMX  =  float(opts[i][1])    

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
    else:
        print(f"Parametro '{opts[i][0][1:]}' desconocido")
        print("Ejecutar con:   python3.9 multi.py -p <paralelo/secuencial> -size <tsplib/Small/Medium/Large>")
        print("Lista de parametros:")
        for i in [["OX","PMX","UPMX"],["NNH","TSP","RPT","NNHJ","RPJ"],["MS1","MS2","EM","RM","SM","OPT2","JLS","JEM"],["POB","CXPB","MUTPB","IT","ELITE","TOURN"]]:
            print("\t",i)
        exit(0)

if size.lower() not in ["tsplib","small","medium","large"]:
    print("Parametro size mal ingresado. (tsplib/small/medium/large)")
    exit(0)

if paralelo.lower() not in ["paralelo","secuencial"]:
    print("Tipo de ejecución mal ingresado. (paralelo/secuencial)")
    exit(0)

if alg.lower() not in ["ag","gurobi"]:
    print("Algoritmo mal ingresado. (ag/gurobi)")
    exit(0)

if alg.lower() == "gurobi" and subtour == None:
    print("Error, ingresar subtour. -subtour <0/1/2/3> (0:Nada / 1:GG / 2: MTZ / 3: DL)")
    exit(0)

if subtour != None and (subtour<0 or subtour >3):
    print("Error, ingresar subtour. -subtour <0/1/2/3> (0:Nada / 1:GG / 2: MTZ / 3: DL)")
    exit(0)

size = size.capitalize()

parameters_value = {"OX": P_OX, "PMX": P_PMX, "UPMX": P_UMPX,  #Cruzamiento
                    "NNH": P_NNH, "TSP": P_TSP, "RPT": P_RPT, "NNHJ": P_NNHJ, "RPJ": P_RPJ,  #Población inicial
                    "MS1": MS1, "MS2": MS2, "EM": P_EM, "RM": P_RM, "SM": P_SM, "OPT2": P_2OPT, "JLS": P_JLS,"JEM":P_JEM, #Mutación
                    "POB":POBLACION , "CXPB":CXPB, "MUTPB":MUTPB, "IT": IT , "ELITE": ELITE, "TOURN":TOURN} #Overall 



if paralelo.lower() == "paralelo":
    if __name__=="__main__":
        seed = [i for i in range(10)]
        pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-2))

        if size.lower() in ["small","medium","large"]:
            if alg == "ag":
                #semilla,size,batch,instancia,mejor,poblacion,tiempo,mejor inicial,poblacion inicial,mejor iteracion,tiempo poblacion
                print("{:<6}{:<8}{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","size","batch","ins","mejor","pob","time","mejor_i","pob_i","mejor_iter","t_pob"))
                instancias = [i for i in range(1,101) ]
                u = 0
                for _product in itertools.product(instancias,seed):
                    _instancia,_seed = _product
                    if _instancia<=25: _batch = 1
                    elif _instancia<=50: _batch = 2
                    elif _instancia<=75: _batch = 3
                    else: _batch = 4
                    pool.apply_async(launcher2, args=(_seed,"paralelo",size,_batch,_instancia),kwds=parameters_value)
            
            else:

                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<20}".format("ins","obj","lb","gap","time","status","solcount"))
                instancias = [i for i in range(1,101) ]
                for instancia in instancias:
                    #print("a")
                    pool.apply_async(launcher3, args=(size,instancia,subtour,True,False,0))
                pool.close()
                pool.join()

        else:
            instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
            if alg == "ag":
                #semilla,instancia,mejor,poblacion,tiempo,mejor inicial, poblacion inicial, iteracion mejor,tiempo_poblacion
                print("{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","ins","mejor","pob","time","mejor_i","pob_i","mejor_iter","t_pob"))
                for _product in itertools.product(instancias,seed):
                    _instancia,_seed = _product
                    pool.apply_async(launcher1, args=(_seed,"paralelo",_instancia) , kwds=parameters_value)
            else:
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<20}".format("ins","obj","lb","gap","time","status","solcount"))
                for instancia in instancias:
                    pool.apply_async(launcher3, args=(size.lower(),instancia,subtour,True,False,0))
                pool.close()
                pool.join()    
        pool.close()
        pool.join()
        #print(time.time()-inicio)

else:
    if size.lower() =="tsplib":
        instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
        #semilla,instancia,mejor,poblacion,tiempo,mejor inicial, poblacion inicial, iteracion mejor,tiempo_poblacion
        if alg == "ag":
            print("{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","ins","mejor","pob","time","mejor_i","pob_i","mejor_iter","t_pob"))
            for i in instancias:
                for j in range(10):
                    launcher1(j,"secuencial",i,OX = P_OX , PMX = P_PMX , UPMX = P_UPMX  , NNH = P_NNH , TSP = P_TSP , RPT = P_RPT , NNHJ = P_NNHJ , RPJ = P_RPJ , MS1 = MS1 , MS2 = MS2 , EM = P_EM , RM = P_RM , SM = P_SM , OPT2 = P_2OPT , JLS = P_JLS , JEM = P_JEM , POB = POBLACION , CXPB = CXPB , MUTPB = MUTPB , IT = IT , ELITE = ELITE , TOURN = TOURN)
        else:
            print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}{}".format("ins","obj","lb","gap","time","status","solcount","Nodos","cortes"))
            for i in instancias:
                launcher3(size.lower(),i,subtour,True,False,0)
    else:
        instancias = [i for i in range(1,101)]
        #semilla,size,batch,instancia,mejor,poblacion,tiempo,mejor inicial,poblacion inicial,mejor iteracion,tiempo poblacion
        if alg == "ag":
            print("{:<6}{:<8}{:<6}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","size","batch","ins","mejor","pob","time","mejor_i","pob_i","mejor_iter","t_pob"))
            for i in instancias:
                for j in range(10):
                    if i<=25: _batch = 1
                    elif i<=50: _batch = 2
                    elif i<=75: _batch = 3
                    else: _batch = 4
                    launcher2(j,"paralelo",size,_batch,i,OX = P_OX , PMX = P_PMX , UPMX = P_UPMX , NNH = P_NNH , TSP = P_TSP , RPT = P_RPT , NNHJ = P_NNHJ , RPJ = P_RPJ , MS1 = MS1 , MS2 = MS2 , EM = P_EM , RM = P_RM , SM = P_SM , OPT2 = P_2OPT , JLS = P_JLS , JEM = P_JEM , POB = POBLACION , CXPB = CXPB , MUTPB = MUTPB , IT = IT , ELITE = ELITE , TOURN = TOURN)
        else:
            print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<20}".format("ins","obj","lb","gap","time","status","solcount"))
            for i in instancias:
                launcher3(size.lower(),i,subtour,True,False,0)
    #print(time.time()-inicio)
