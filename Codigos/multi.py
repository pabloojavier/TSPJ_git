import itertools
import os
import multiprocessing
import time
import random
import sys



inicio = time.time()
def launcher1(semilla,paralelo,_instancia , CRZ1=0.4 , CRZ2=0.2 , CRZ3=0.3 , CRZ4=0.2 , VMC1=0.9  , CT1=0.3 , MS1=0.5 , MS11=0.25 , MS12=0.25 , MS13=0.25 , MS14=0.25 , MS21=0.3 , POB=50 , CXPB=0.9 , MUTPB=0.2 , IT=500,ELITE=0.1,TOURN=4):
    os.system(f"python3.9 ga_05.py -paralelo {paralelo} -seed {semilla} -i {_instancia} -CRZ1 {CRZ1} -CRZ2 {CRZ2} -CRZ3 {CRZ3} -CRZ4 {CRZ4} -VMC1 {VMC1} -CT1 {CT1} -MS1 {MS1} -MS11 {MS11} -MS12 {MS12} -MS13 {MS13} -MS14 {MS14} -MS21 {MS21} -POB {POB} -CXPB {CXPB} -MUTPB {MUTPB} -IT {IT} -ELIT {ELITE} -TOURN {TOURN} -multi True")

def launcher2(semilla,paralelo,_size,_batch,_instancia , CRZ1=0.4 , CRZ2=0.2 , CRZ3=0.3 , CRZ4=0.2 , VMC1=0.9  , CT1=0.3 , MS1=0.5 , MS11=0.25 , MS12=0.25 , MS13=0.25 , MS14=0.25 , MS21=0.3 , POB=50 , CXPB=0.9 , MUTPB=0.2 , IT=500,ELITE=0.1,TOURN=4):
    #print(f"python3.9 ga_05.py -paralelo {paralelo} -seed {semilla} -size {_size} -batch {_batch} -i {_instancia} -CRZ1 {CRZ1} -CRZ2 {CRZ2} -CRZ3 {CRZ3} -CRZ4 {CRZ4} -VMC1 {VMC1} -CT1 {CT1} -MS1 {MS1} -MS11 {MS11} -MS12 {MS12} -MS13 {MS13} -MS14 {MS14} -MS21 {MS21} -POB {POB} -CXPB {CXPB} -MUTPB {MUTPB} -IT {IT} -ELIT {ELITE} -TOURN {TOURN} -multi True")
    os.system(f"python3.9 ga_05.py -paralelo {paralelo} -seed {semilla} -size {_size} -batch {_batch} -i {_instancia} -CRZ1 {CRZ1} -CRZ2 {CRZ2} -CRZ3 {CRZ3} -CRZ4 {CRZ4} -VMC1 {VMC1} -CT1 {CT1} -MS1 {MS1} -MS11 {MS11} -MS12 {MS12} -MS13 {MS13} -MS14 {MS14} -MS21 {MS21} -POB {POB} -CXPB {CXPB} -MUTPB {MUTPB} -IT {IT} -ELIT {ELITE} -TOURN {TOURN} -multi True")

#region
paralelo = "paralelo"
tipo = "pruebas"
size = "Medium"
batch = 1
instancia = "1"
CRZ1      = 0.4
CRZ2      = 0.2
CRZ3      = 0.2
CRZ4      = 0.2
VMC1      = 0.9
CT1       = 0.3
MS1       = 0.5
MS1_1     = 0.25
MS1_2     = 0.25
MS1_3     = 0.25
MS1_4     = 0.25
MS2_1     = 0.3
POBLACION = 50
CXPB      = 0.9
MUTPB     = 0.2
IT        = 500
ELITE     = 0.1
TOURN     = 4
tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]


for i in range(len(opts)):
    if opts[i][0][1:] == "paralelo": paralelo = opts[i][1]
    elif opts[i][0][1:] == "tipo": tipo = opts[i][1] 
    elif opts[i][0][1:] == "size": size = str(opts[i][1]) 
    elif opts[i][0][1:] == "batch" : batch = int(opts[i][1]) 
    elif opts[i][0][1:] == "i":
        try:
            instancia = int(opts[i][1])  
        except ValueError:
            instancia = opts[i][1]
    #elif opts[i][0][1:] == "insttsp": instanciatsp = opts[i][1] 
    elif opts[i][0][1:] == "CRZ1":CRZ1  =  float(opts[i][1])  
    elif opts[i][0][1:] == "CRZ2":CRZ2  =  float(opts[i][1])  
    elif opts[i][0][1:] == "CRZ3":CRZ3  =  float(opts[i][1])  
    elif opts[i][0][1:] == "CRZ4":CRZ4  =  float(opts[i][1])  
    elif opts[i][0][1:] == "VMC1":VMC1  =  float(opts[i][1])   
    elif opts[i][0][1:] == "CT1" :CT1   =  float(opts[i][1]) 
    elif opts[i][0][1:] == "MS1" :MS1   =  float(opts[i][1]) 
    elif opts[i][0][1:] == "MS11":MS1_1 =  float(opts[i][1])  
    elif opts[i][0][1:] == "MS12":MS1_2 =  float(opts[i][1])  
    elif opts[i][0][1:] == "MS13":MS1_3 =  float(opts[i][1])  
    elif opts[i][0][1:] == "MS14":MS1_4 =  float(opts[i][1])   
    elif opts[i][0][1:] == "MS21":MS2_1 =  float(opts[i][1])  
    elif opts[i][0][1:] == "POB": POBLACION = int(opts[i][1]) 
    elif opts[i][0][1:] == "CXPB": CXPB  = float(opts[i][1]) 
    elif opts[i][0][1:] == "MUTPB": MUTPB = float(opts[i][1])
    elif opts[i][0][1:] == "IT": IT = int(opts[i][1])
    elif opts[i][0][1:] == "ELIT": ELITE = float(opts[i][1])
    elif opts[i][0][1:] == "TOURN": TOURN = int(opts[i][1])

#endregion

parameters_value = {"CRZ1": CRZ1, "CRZ2": CRZ2, "CRZ3": CRZ3, "CRZ4": CRZ4, "VMC1": VMC1, "CT1": CT1, "MS1": MS1, "MS11": MS1_1, "MS12": MS1_2, "MS13": MS1_3, "MS14": MS1_4, "MS21": MS2_1, "POB": POBLACION, "CXPB": CXPB, "MUTPB": MUTPB, "IT": IT}


if paralelo.lower() == "paralelo":
    if __name__=="__main__":
        seed = [i for i in range(10)]
        pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-2))

        if tipo.lower() !="tsplib":
            #print("{:<6}{:<8}{:<6}{:<12}{:<12}{:<10}{}".format("seed","size","batch","instancia ","min","avg","tiempo"))
            #print("--------------------------------------------------------------")
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
            #print("{:<6}{:<12}{:<12}{:<10}{}".format("seed","instancia ","min","avg","tiempo"))
            #print("--------------------------------------------------")
            instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
            for _product in itertools.product(instancias,seed):
                _instancia,_seed = _product
                pool.apply_async(launcher1, args=(_seed,"paralelo",_instancia) , kwds=parameters_value)
        
        pool.close()
        pool.join()
        #print(time.time()-inicio)

else:
    if tipo.lower() =="tsplib":
        instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
        #print("{:<6}{:<12}{:<12}{:<10}{}".format("seed","instancia ","min","avg","tiempo"))
        #print("--------------------------------------------------")
        for i in instancias:
            for j in range(10):
                launcher1(j,"secuencial",i,CRZ1=CRZ1 , CRZ2=CRZ2 , CRZ3=CRZ3 , CRZ4=CRZ4 , VMC1=VMC1  , CT1=CT1 , MS1=MS1 , MS11=MS1_1 , MS12=MS1_2 , MS13=MS1_3 , MS14=MS1_4 , MS21=MS2_1 , POB=POBLACION , CXPB=CXPB , MUTPB=MUTPB , IT=IT,ELITE=ELITE,TOURN=TOURN)
    else:
        instancias = [i for i in range(1,101) ]
        #print("{:<6}{:<8}{:<6}{:<12}{:<12}{:<10}{}".format("seed","size","batch","instancia ","min","avg","tiempo"))
        #print("--------------------------------------------------------------")
        for i in instancias:
            for j in range(10):
                if i<=25: _batch = 1
                elif i<=50: _batch = 2
                elif i<=75: _batch = 3
                else: _batch = 4
                launcher2(j,"paralelo",size,_batch,i,CRZ1=CRZ1 , CRZ2=CRZ2 , CRZ3=CRZ3 , CRZ4=CRZ4 , VMC1=VMC1  , CT1=CT1 , MS1=MS1 , MS11=MS1_1 , MS12=MS1_2 , MS13=MS1_3 , MS14=MS1_4 , MS21=MS2_1 , POB=POBLACION , CXPB=CXPB , MUTPB=MUTPB , IT=IT,ELITE=ELITE,TOURN=TOURN)
    #print(time.time()-inicio)
