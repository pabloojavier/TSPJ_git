import itertools
import os
import multiprocessing
import time



path = "Codigos/"
inicio = time.time()
def launcherMGA(seed,size,_instancia,**kwargs):
    ejecutar = f"/usr/local/bin/python3 {path}ga_05.py -seed {seed} -size {size} -i {_instancia}"
    for key,value in kwargs.items():
        ejecutar += f"-{key} {value} "
    os.system(ejecutar)

def launcherMILP(size,instance,subtour,solin,output,callback,bounds):
    os.system(f"/usr/local/bin/python3 {path}MM_poo.py -size {size} -instance {instance} -subtour {subtour} -initialsol {solin} -output {output} -callback {callback} -bounds {bounds}")

#Valores por defecto
paralelo = "paralelo"
size = "tsplib"
alg = "gurobi"
subtour = "subtourelim"


tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
sml_instance = [i for i in range(1,101)]
instance_dict = {"tsplib":tsplib,
                 "small":sml_instance,
                 "medium":sml_instance,
                 "large":sml_instance}


argv = ["-p","secuential","-size","tsplib","-alg","gurobi","-subtour","GG","-initialsol","True", "-callback","None","-bounds","False"]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

if len(opts)<3:
    print("multi.py -p <parallel/secuential> -size <tsplib/small/medium/large/all> -alg <ag/gurobi> -subtour <wc/gg/mtz/dl> -initialsol <True/False> -callback <*callbacklist*> -bounds <True/False>"  )
    exit(0)

if len(argv)%2 != 0:
    print("Error, falta un parametro o valor")
    exit(0)

for i in range(len(opts)):
    if opts[i][0][1:] == "p": paralelo = opts[i][1].lower()
    elif opts[i][0][1:] == "size": size = str(opts[i][1]).lower()
    elif opts[i][0][1:] == "alg": alg = str(opts[i][1]).lower()
    elif opts[i][0][1:] == "subtour": subtour = str(opts[i][1]).lower()
    elif opts[i][0][1:] == "initialsol": initialsol = str(opts[i][1]).lower()
    elif opts[i][0][1:] == "callback": callback = str(opts[i][1])
    elif opts[i][0][1:] == "bounds": bounds = str(opts[i][1]).lower()
    # elif opts[i][0][1:] == "OX"   : P_OX    =  float(opts[i][1])
    # elif opts[i][0][1:] == "PMX"  : P_PMX   =  float(opts[i][1])  
    # elif opts[i][0][1:] == "UMPX" : P_UPMX  =  float(opts[i][1])    

    # elif opts[i][0][1:] == "NNH"  : P_NNH   =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "TSP"  : P_TSP   =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "RPT"  : P_RPT   =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "NNHJ" : P_NNHJ  =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "RPJ"  : P_RPJ   =  float(opts[i][1]) 

    # elif opts[i][0][1:] == "MS1"  : MS1     =  float(opts[i][1])
    # elif opts[i][0][1:] == "MS2"  : MS2     =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "EM"   : P_EM    =  float(opts[i][1])  
    # elif opts[i][0][1:] == "RM"   : P_RM    =  float(opts[i][1])  
    # elif opts[i][0][1:] == "SM"   : P_SM    =  float(opts[i][1]) 
    # elif opts[i][0][1:] == "OPT2" : P_2OPT  =  float(opts[i][1])   
    # elif opts[i][0][1:] == "JLS"  : P_JLS   =  float(opts[i][1])  
    # elif opts[i][0][1:] == "JEM"  : P_JEM   =  float(opts[i][1])  

    # elif opts[i][0][1:] == "POB": POBLACION = int(opts[i][1]) 
    # elif opts[i][0][1:] == "CXPB" : CXPB   = float(opts[i][1]) 
    # elif opts[i][0][1:] == "MUTPB": MUTPB  = float(opts[i][1])
    # elif opts[i][0][1:] == "IT"   : IT     = int(opts[i][1])
    # elif opts[i][0][1:] == "ELITE" : ELITE  = float(opts[i][1])
    # elif opts[i][0][1:] == "TOURN": TOURN  = int(opts[i][1])
    else:
        print(f"Parameter :'{opts[i][0][1:]}' unknown")
        print("Execute with: python3.9 multi.py -p <parallel/secuential> -size <tsplib/Small/Medium/Large>")
        print("Parameter list:")
        for i in [["OX","PMX","UPMX"],["NNH","TSP","RPT","NNHJ","RPJ"],["MS1","MS2","EM","RM","SM","OPT2","JLS","JEM"],["POB","CXPB","MUTPB","IT","ELITE","TOURN"]]:
            print("\t",i)
        exit(0)

if size not in ["tsplib","small","medium","large"]:
    print("Size parameter error. (tsplib/small/medium/large)")
    exit(0)

if paralelo not in ["parallel","secuential"]:
    print("Execution type error. (parallel/secuential)")
    exit(0)

if alg not in ["mga","gurobi"]:
    print("Algorithm error. (mga/gurobi)")
    exit(0)

if alg == "gurobi" and (subtour not in ("wc","gg","mtz","dl")) :
    print("Error, ingresar subtour. -subtour <WC/GG/MTZ/DL>")
    exit(0)

# parameters_value = {"OX": P_OX, "PMX": P_PMX, "UPMX": P_UPMX,  #Cruzamiento
#                     "NNH": P_NNH, "TSP": P_TSP, "RPT": P_RPT, "NNHJ": P_NNHJ, "RPJ": P_RPJ,  #Población inicial
#                     "MS1": MS1, "MS2": MS2, "EM": P_EM, "RM": P_RM, "SM": P_SM, "OPT2": P_2OPT, "JLS": P_JLS,"JEM":P_JEM, #Mutación
#                     "POB":POBLACION , "CXPB":CXPB, "MUTPB":MUTPB, "IT": IT , "ELITE": ELITE, "TOURN":TOURN} #Overall 


if __name__ == "__main__":
    seed = [i for i in range(10)]
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-2))

    if alg == "mga":
        print("{:<6}{:<8}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format("seed","size","instance","mejor","pob","time","mejor_initi","pob_i","mejor_iter","t_pob"))
        for _instancia,_seed in itertools.product(instance_dict[size],seed):
            if paralelo == "parallel":
                pool.apply_async(launcherMGA, args=(_seed,size,_instancia),kwds={})
            elif paralelo == "secuential":
                launcherMGA(_seed,size,_instancia)
    else:
        print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}{:<10}{:<10}".format("size","instance","obj","lb","gap","time","status","Sol Count","NodeCount","#callback","timecallback"))
        for _instance in instance_dict[size]:
            if paralelo == "parallel":
                pool.apply_async(launcherMILP, args=(size,_instance,subtour,initialsol,"False",callback,bounds))
            elif paralelo == "secuential":
                launcherMILP(size,_instance,subtour,initialsol,"False",callback,bounds)
    
    if paralelo == "parallel":
        pool.close()
        pool.join()
    




