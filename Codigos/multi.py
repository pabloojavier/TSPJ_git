import itertools
import os
import multiprocessing
import time
import sys

def launcherMGA(seed,size,_instancia,**kwargs):
    ejecutar = f"/usr/local/bin/python3 {path}MGA_poo.py -seed {seed} -size {size} -i {_instancia} "
    for key,value in kwargs.items():
        ejecutar += f"-{key} {value} "
    os.system(ejecutar)

def launcherMILP(size,instance,subtour,solin,output,callback,bounds,_new_formulation):
    os.system(f"/usr/local/bin/python3 {path}MM_poo.py -size {size} -instance {instance} -subtour {subtour} -initialsol {solin} -output {output} -callback {callback} -bounds {bounds} -newformulation {_new_formulation}")

def set_parameters(argv):
    global paralelo,size,alg,subtour,bounds,initialsol,callback,new_formulation, P_OX, P_PMX, P_UPMX, P_NNH, P_TSP, P_RPT, P_NNHJ, P_RPJ, MS1, MS2, P_EM, P_RM, P_SM, P_2OPT, P_JLS, P_JEM, ELITE, POBLACION, CXPB, MUTPB, IT, TOURN, TIMELIMIT,parameters_value
    paralelo = "secuential"
    size = "tsplib"
    alg = "gurobi"
    subtour = "gg"
    initialsol = "True"
    bounds = 'True'
    callback = 'none'
    new_formulation = 'True'
    P_RPT = 0.236 #0.1       
    P_NNH = 0.112 #0.5 
    P_TSP = 0.652 #0.4
    P_RPJ = 0.497 #0.3
    P_NNHJ = 0.503 #0.7
    P_OX = 0.429400386847195 #0.5
    P_PMX = 0.362669245647969 #0.25
    P_UPMX = 0.207930367504836  #0.25
    MS1 = 0.121  #0.5
    MS2 = 0.883  #0.5
    P_EM = 0.338612779901693  #0.25 
    P_RM = 0.252867285636264 #0.25 
    P_SM = 0.395412342981977 #0.25 
    P_2OPT = 0.0131075914800655 #0.25 
    P_JLS = 0.914 #0.3
    P_JEM = 0.086 #0.7 #Exchange mutation job, JLS complement
    ELITE = 0.147 #0.1
    POBLACION = 100 #50
    CXPB = 0.327  #0.9
    MUTPB = 0.717  #0.2
    IT = 500 #500
    TOURN = 2 #4s
    TIMELIMIT = 1800 #1800

    opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

    if len(opts)<3:
        print("multi.py -p <parallel/secuential> -size <tsplib/small/medium/large/all> -alg <mga/gurobi> -subtour <wc/gg/mtz/dl> -initialsol <True/False> -callback <*callbacklist*> -bounds <True/False>"  )
        raise ValueError("Error, faltan parametros")

    if len(argv)%2 != 0:
        raise ValueError("Error, falta un parametro o valor")

    for i in range(len(opts)):
        if opts[i][0][1:] == "p": paralelo = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "size": size = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "alg": alg = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "subtour": subtour = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "initialsol": initialsol = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "callback": callback = str(opts[i][1])
        elif opts[i][0][1:] == "bounds": bounds = str(opts[i][1]).lower()
        elif opts[i][0][1:] == "newformulation": new_formulation = str(opts[i][1]).lower()
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
        elif opts[i][0][1:] == "POB"  : POBLACION = int(opts[i][1]) 
        elif opts[i][0][1:] == "CXPB" : CXPB   = float(opts[i][1]) 
        elif opts[i][0][1:] == "MUTPB": MUTPB  = float(opts[i][1])
        elif opts[i][0][1:] == "IT"   : IT     = int(opts[i][1])
        elif opts[i][0][1:] == "ELITE": ELITE  = float(opts[i][1])
        elif opts[i][0][1:] == "TOURN": TOURN  = int(opts[i][1])
        elif opts[i][0][1:] == "TIMELIMIT": TIMELIMIT  = int(opts[i][1])
        else:
            print(f"Parameter :'{opts[i][0][1:]}' unknown")
            print("Execute with: python3.9 multi.py -p <parallel/secuential> -size <tsplib/Small/Medium/Large>")
            print("Parameter list:")
            for i in [["p",'size','alg','subtour','initialsol','callback','bounds','newformulation'],
                      ["P_OX","P_PMX","P_UPMX"],
                      ["P_NNH","P_TSP","P_RPT","P_NNHJ","P_RPJ"],
                      ["MS1","MS2","P_EM","P_RM","P_SM","P_2OPT","P_JLS","P_JEM"],
                      ["POB","CXPB","MUTPB","IT","ELITE","TOURN"]]:
                print("\t",i)
            raise ValueError(f"Parameter :'{opts[i][0]}' unknown")

    if (    size     not in ["tsplib","small","medium","large"] 
        or  paralelo not in ["parallel","secuential"] 
        or  alg      not in ["mga","gurobi"] 
        or (alg == "gurobi" and (subtour not in ("wc","gg","mtz","dl","dl_real")))):
        raise ValueError(f"Error, parametro desconocido ({size}/{paralelo}/{alg}/{subtour}))")

    parameters_value = {"P_OX": P_OX, "P_PMX": P_PMX, "P_UPMX": P_UPMX,  #Cruzamiento
                        "P_NNH": P_NNH, "P_TSP": P_TSP, "P_RPT": P_RPT, "P_NNHJ": P_NNHJ, "P_RPJ": P_RPJ,  #Población inicial
                        "MS1": MS1, "MS2": MS2, "P_EM": P_EM, "P_RM": P_RM, "P_SM": P_SM, "P_2OPT": P_2OPT, "P_JLS": P_JLS,"P_JEM":P_JEM, #Mutación
                        "POPULATION":POBLACION , "CXPB":CXPB, "MUTPB":MUTPB, "IT": IT , "ELITE": ELITE, "TOURN":TOURN, "CXPB":CXPB,"MUTPB":MUTPB,"TIMELIMIT":TIMELIMIT} #Overall 

inicio = time.time()
if os.getcwd().split("/")[-1] != "Codigos":
    path = "Codigos/"
else:
    path = ""

tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
sml_instance = [i for i in range(1,101)]
instance_dict = {"tsplib":tsplib,
                 "small":sml_instance,
                 "medium":sml_instance,
                 "large":sml_instance}

argv = ["-p"              , "secuential",
        "-size"           , "tsplib",
        "-alg"            , "gurobi",
        "-subtour"        , "wc",
        "-initialsol"     , "False",
        "-callback"       , "none",
        "-bounds"         , "False",
        "-newformulation" , "True"]

#argv = sys.argv[1:]
set_parameters(argv)

if __name__ == "__main__":
    seed = [i for i in range(10)]
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-2))
    print(f"Running with parameters: {[i for i in argv]} ")
    if alg == "mga":
        print("{:<6}{:<8}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","size","instance","mejor","pob","time","mejor_initi","pob_i","mejor_iter","t_pob"))
        for _instancia,_seed in itertools.product(instance_dict[size],seed):
            if paralelo == "parallel":
                pool.apply_async(launcherMGA, args=(_seed,size,_instancia),kwds={parameters_value})
            elif paralelo == "secuential":
                launcherMGA(_seed,size,_instancia,**parameters_value)
    else:
        print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}{:<10}{:<10}".format("size","instance","obj","lb","gap","time","status","Sol Count","NodeCount","#callback","timecallback"))
        for _instance in instance_dict[size]:
            if paralelo == "parallel":
                pool.apply_async(launcherMILP, args=(size,_instance,subtour,initialsol,"False",callback,bounds,new_formulation))
            elif paralelo == "secuential":
                launcherMILP(size,_instance,subtour,initialsol,"False",callback,bounds,new_formulation)
    
    if paralelo == "parallel":
        pool.close()
        pool.join()
    




