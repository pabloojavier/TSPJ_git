from Source.MGA import MGA
import sys

argv = sys.argv[1:]
opts = [(argv[2*i],argv[2*i+1]) for i in range(int(len(argv)/2))]

size = "tsplib"
instance = "gr17"
seed = 0

for i in range(len(opts)):
    if opts[i][0][1:] == "size":  size  = opts[i][1]
    elif opts[i][0][1:] == "i": 
        try:
            instance = int(opts[i][1])  
        except ValueError:
            instance = opts[i][1]  
    elif   opts[i][0][1:] == "seed": seed = int(opts[i][1])

mga = MGA(size,instance,seed = seed)
mga.compare = True
mga.run()
mga.print_results()
#print(mga.get_solution())
#print(mga.fitness_functions(mga.get_solution()))
