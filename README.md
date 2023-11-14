# A multioperator Genetic Algorithm for the Traveling Salesman Problem with Job-times (TSPJ)
date: "November, 2023"

## Abstract:

This paper addresses the traveling salesman problem with job-times (TSPJ). TSPJ considers two sets of equal size, a set of tasks and a set of vertices, where a traveler must visit each vertex exactly once, return to the starting vertex, and perform a unique task at each vertex. Each task is assigned to only one vertex, and each task is completed with a job-time that depends on each vertex. Thus, the objective is to minimize the time of the last task performed. However, due to its NP-hardness, existing algorithms do not optimally solve the TSPJ larger instances. Therefore, we propose an approach based on a multioperator genetic algorithm (MGA) that utilizes various initial population procedures, crossover and mutation operators. MGA applies five initial population procedures to generate a diverse population, and employs three crossover operators and six mutation operators, with four mutations focused on diversification and two designed to aid intensification. Furthermore, to improve the quality of the best individual found, a local search is used in every generation, and generational replacement with elitism is considered when generating a new population. MGA is evaluated on four sets of instances ranging in size from 17 to 1200 vertices: tsplib, small, medium, and large. Our approach outperforms the state-of-the-art algorithms on the four instance sets. This performance is attributed to the synergistic effect of the multioperators of crossover and mutation, along with effective parameter tuning.


## How to run
* All codes are in 'Codigos' folder.
* Just run 'MGA_po.py' file to run just one instance
* To run all instance from one size, run 'multi.py'

### Args to run
To run 'multi.py' you have to specify the following args:

```
python3.9 multi.py -p <parallel/secuential> -size <tsplib/Small/Medium/Large>
```

Where 'p' is the type of execution (parallel or secuential) and 'size' is the size of the instances to run.

Also, you can change the following args (OPTIONAL):
- OX
- PMX
- UMPX
- NNH
- TSP
- RPT
- NNHJ
- RPJ
- MS1
- MS2
- EM
- RM
- SM
- OPT2
- JLS
- JEM
- POB
- CXPB
- MUTPB
- IT
- ELITE
- TOURN
- TIMELIMIT

If you dont change them, the default values will be used.
An example of how to change them is:
```
python3 multi.py -p parallel -size tsplib -OPT2 0.9 -TIMELIMIT 100 -IT 1000
```
