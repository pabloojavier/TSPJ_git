from Source.MGA import MGA
import sys

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

size = "tsplib"
instance = "gr17"
seed = 0
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
TIMELIMIT = 1800

for i in range(len(opts)):
    if opts[i][0][1:] == "size":  size  = opts[i][1]
    elif opts[i][0][1:] == "i": 
        try:
            instance = int(opts[i][1])  
        except ValueError:
            instance = opts[i][1]  
    elif   opts[i][0][1:] == "seed": seed = int(opts[i][1])
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

parameters_value = {"P_OX": P_OX, "P_PMX": P_PMX, "P_UPMX": P_UPMX,  #Cruzamiento
                    "P_NNH": P_NNH, "P_TSP": P_TSP, "P_RPT": P_RPT, "P_NNHJ": P_NNHJ, "P_RPJ": P_RPJ,  #Población inicial
                    "MS1": MS1, "MS2": MS2, "P_EM": P_EM, "P_RM": P_RM, "P_SM": P_SM, "P_2OPT": P_2OPT, "P_JLS": P_JLS,"P_JEM":P_JEM, #Mutación
                    "POPULATION":POBLACION , "CXPB":CXPB, "MUTPB":MUTPB, "IT": IT , "ELITE": ELITE, "TOURN":TOURN, "CXPB":CXPB,"MUTPB":MUTPB,"TIMELIMIT":TIMELIMIT} #Overall 

mga = MGA(size,instance,seed = seed)
mga.parameters_value = parameters_value
mga.compare = True
mga.run()
mga.print_results()
#print(mga.get_solution())
#print(mga.fitness_functions(mga.get_solution()))
