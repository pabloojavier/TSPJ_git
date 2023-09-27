#!/usr/bin/env python3.9
from operator import index
from numpy.random.mtrand import rand, seed
import matplotlib.pyplot as plt
import random
import time
#import animated_visualizer as animacion
import pandas as pd
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from itertools import combinations

import gurobipy as gp
from gurobipy import *

from typing import Dict, List, Callable, Any
from Source.Problem import Problem
from Source.MathematicalModel import MathematicalModel as MM

class MGA(Problem):
    """
    To run it:
    >>> mga = MGA(size,instance)
    >>> mga.run()
    >>> solution = mga.get_solution()
    """
    def __init__(self,
                 size,
                 instance,
                 compare=True,
                 seed = 0):
        super().__init__(size,instance)
        self.parameters = {"P_RPT"      : 0.236, #0.1       
                          "P_NNH"      : 0.112, #0.5 
                          "P_TSP"      : 0.652, #0.4
                          "P_RPJ"      : 0.497, #0.3
                          "P_NNHJ"     : 0.503, #0.7
                          "P_OX"       : 0.429400386847195, #0.5
                          "P_PMX"      : 0.362669245647969, #0.25
                          "P_UMPX"     : 0.207930367504836,  #0.25
                          "MS1"        : 0.121,  #0.5
                          "MS2"        : 0.883,  #0.5
                          "P_EM"       : 0.338612779901693,  #0.25 
                          "P_RM"       : 0.252867285636264, #0.25 
                          "P_SM"       : 0.395412342981977, #0.25 
                          "P_2OPT"     : 0.0131075914800655, #0.25 
                          "P_JLS"      : 0.914, #0.3
                          "P_JEM"      : 0.086, #0.7 #Exchange mutation job, JLS complement
                          "ELITE"      : 0.147, #0.1
                          "POPULATION" : 100, #50
                          "CXPB"       : 0.327,  #0.9
                          "MUTPB"      : 0.717,  #0.2
                          "IT"         : 500, #500
                          "TOURN"      : 2, #4s
                          "TIMELIMIT"  : 1800
                          }
        self.g = 0
        self.best_iteration = 0 
        self.compare = compare
        self.seed = seed
        self.FlagRoute = False

    def NearestNeighborAlgorithm(self,n):
        _from = random.randint(1,n)
        actual = _from
        route = []
        route.append(_from)
        selectioned = [False] * n
        selectioned[actual-1] = True
        while len(route) < n:
            min = float("inf")
            for candidate in range(1,n+1):
                if selectioned[candidate-1] == False and candidate != actual:
                    cost = self.TT[actual][candidate]
                    if cost < min:
                        min = cost
                        _next = candidate
            route.append(_next)
            selectioned[_next-1] = True
            actual = _next
        return route

    def __sort_solution(self,list,n):
        actual = 0
        _solution = []
        while len(_solution) < n+1:
            _solution.append(actual)
            for i in list:
                if i[0]==actual and i[1] not in _solution:
                    _next = i[1]
                    actual = _next
                    break
        return _solution

    @staticmethod
    def subtourelim1(model:gp.Model, where):
        def subtour(subruta, vals,n):
            arcos = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
            noVisitados = list(range(n))
            while noVisitados: 
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
        
        n = model._N
        if where == GRB.Callback.MIPSOL:
            valoresX = model.cbGetSolution(model._vars)
            # encontrar ciclo más pequeño
            tour = [i for i in range(n+1)]
            subtour(tour, valoresX,n)
            if len(tour) < n:
                model.cbLazy(gp.quicksum(model._vars[i, j]for i, j in combinations(tour, 2))<= len(tour)-1)

    def TSPGurobi(self,n,time=10):

        nodes = [i for i in range(n+1)]
        arch = [(i, j) for i in nodes for j in nodes if i < j]
        dist = {(i, j):self.TT[i][j] for i, j in arch}

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
                for i, j in x.keys():
                    x[j, i] = x[i, j]
                model.addConstrs(x.sum(i, '*') == 2 for i in range(n+1))
                model.Params.Threads = 1
                model._vars = x
                model._N = self.n
                model.Params.LazyConstraints = 1
                model.setParam('TimeLimit', time)
                model.optimize(MGA.subtourelim1)
                solucion = []
                for i in x:
                    if x[i].X >0.8:
                        solucion.append(i)
                solucion = self.__sort_solution(solucion,n)
        return solucion[1:]

    def generateRoute(self,n):
        aleatorio =random.uniform(0, 1) 
        P_NNH = self.parameters["P_NNH"]
        P_TSP = self.parameters["P_TSP"]

        if aleatorio < P_NNH:
            route = self.NearestNeighborAlgorithm(n)

        elif aleatorio <P_NNH+P_TSP:
            if self.FlagRoute==False:
                if n<150:
                    route = self.TSPGurobi(n,2000)
                else:
                    route = self.get_lkh_route()
                self.gurobi_route=route.copy()
                self.FlagRoute=True
            else:
                route = self.gurobi_route
        else:
            route = [i for i in range(1, n+1)]
            random.shuffle(route)
        return route
    
    def generateJobs(self,route,jobs):
        value = random.uniform(0,1)

        if value < 1-self.parameters["P_NNHJ"]:
            n = len(route)
            jobs = [i for i in range(1,n+1)]
            random.shuffle(jobs)
        else:
            jobs = MM.NNJA(route,self.JT)
        return jobs

    def initial_population(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.fitness_functions)
        t_aux = time.time()
        self.toolbox.register("cromo1",self.generateRoute,self.n-1)
        self.toolbox.register("cromo2", lambda x:None, self.n)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,(self.toolbox.cromo1, self.toolbox.cromo2), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.pop = self.toolbox.population(n=self.parameters["POPULATION"])
        for i in self.pop:
            i[1] = self.generateJobs(i[0],i[1]) 
        self.fitnesses = list(map(self.toolbox.evaluate, self.pop))
        self.initial_pop_fitness = numpy.mean(self.fitnesses)
        self.best_initial = numpy.min(self.fitnesses)
        self.best_aux = self.best_initial

        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit
        self.population_time = time.time()-t_aux

    def initialize(self):
        self.toolbox.register("mate",self.crossover_main_operator)
        self.toolbox.register("mutate", self.mutSet)
        self.toolbox.register("select", tools.selTournament, tournsize=self.parameters["TOURN"])
        self.elite_size = self.parameters["POPULATION"]*self.parameters["ELITE"]
        self.hof = tools.HallOfFame(self.elite_size)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        self.stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)
        self.log = tools.Logbook()
        self.log.header = "gen", "evals", "std", "min", "avg", "max"
        self.record = self.stats.compile(self.pop)
        self.log.record(gen=self.g, evals=len(self.pop), **self.record)
        self.hof.update(self.pop)

    def crossover(self,offspring):
        offspring = list(map(self.toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.parameters["CXPB"]:
                self.toolbox.mate(child1[0], child2[0])
                self.toolbox.mate(child1[1], child2[1])
                del child1.fitness.values
                del child2.fitness.values
        return offspring

    def OX(self,ind1:List,ind2:List):
        """
        This crossover generates holes in the input
        individuals. A hole is created when an attribute of an individual is
        between the two crossover points of the other individual. Then it rotates
        the element so that all holes are between the crossover points and fills
        them with the removed elements in order. More details in Goldberg1989
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

        temp1, temp2 = ind1, ind2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                ind1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % size]]:
                ind2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1

        for i in range(a, b + 1):
            ind1[i], ind2[i] = ind2[i], ind1[i]

        for i in range(size): 
            ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

    def PMX(self,ind1:List, ind2:List):
        """
        Moreover, this crossover generates two children by matching
        pairs of values in a certain range of the two parents and swapping the values
        of those indexes. For more details see [Goldberg1985]
        """
        size = min(len(ind1), len(ind2))
        p1, p2 = [0] * size, [0] * size

        for i in range(size):
            ind1[i],ind2[i] = ind1[i]-1,ind2[i]-1

        for i in range(size):
            p1[ind1[i]] = i
            p2[ind2[i]] = i
        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        for i in range(cxpoint1, cxpoint2):
            temp1 = ind1[i]
            temp2 = ind2[i]
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        for i in range(size): 
            ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

    def UMPX(self,ind1, ind2, indpb=0.5):
        """
        This crossover generates two children by matching
        pairs of values chosen at random with a probability of *indpb* in the two
        parents and swapping the values of those indexes. For more details see
        Cicirello2000.
        """
        size = min(len(ind1), len(ind2))
        p1, p2 = [0] * size, [0] * size

        for i in range(size):
            ind1[i],ind2[i] = ind1[i]-1,ind2[i]-1
        
        for i in range(size):
            p1[ind1[i]] = i
            p2[ind2[i]] = i

        for i in range(size):
            if random.random() < indpb:
                temp1 = ind1[i]
                temp2 = ind2[i]
                ind1[i], ind1[p1[temp2]] = temp2, temp1
                ind2[i], ind2[p2[temp1]] = temp1, temp2
                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
        for i in range(size):
            ind1[i],ind2[i] = ind1[i]+1,ind2[i]+1

    def crossover_main_operator(self,ind1,ind2):
        value = random.uniform(0,1)
        P_OX = self.parameters["P_OX"]
        P_PMX = self.parameters["P_PMX"]
        P_UMPX = self.parameters["P_UMPX"]
        if value <P_OX:
            self.OX(ind1,ind2)
        elif value <P_OX+P_PMX:
            self.PMX(ind1,ind2)
        elif value<P_OX+P_PMX+P_UMPX:
            self.UMPX(ind1,ind2)

    def mutation(self,offspring):
        for mutant in offspring:
            if random.random() < self.parameters["MUTPB"]:
                self.toolbox.mutate(mutant[0],mutant[1])
                del mutant.fitness.values  
        return offspring
            
    def ExchangeMutation(self,city):
        i = 0
        j = 0
        n = len(city)
        while i == j:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
        temp = city[i]
        city[i] = city[j]
        city[j] = temp

    def ReverseMutation(self,city):
        i = 0
        j = 0
        n = len(city)
        while i >= j:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
        city[i : j] = city[i : j][::-1]

    def SuccessorMutation(self,city):
        i = 0
        j = 0
        n = len(city)

        while i == j:
            i = random.randint(0, n - 2)
            # j = random.randint(0, n - 1)
        j = i + 1
            # intercambio
        temp = city[i]
        city[i] = city[j]
        city[j] = temp

    def DosOpt(self,city):
        city.insert(0,0)
        actual = 0
        n = len(city)
        flag = True
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                nuevoCosto = self.TT[city[i]][city[j]] + self.TT[city[i + 1]][city[j + 1]] - self.TT[city[i]][city[i+1]] - self.TT[city[j]][city[j+1]]
                if nuevoCosto < actual:
                    actual = nuevoCosto
                    min_i, min_j = i, j
                    if flag == True:
                        flag = False
            if flag == False:
                break

        if actual < 0:
            city[min_i + 1 : min_j + 1] = city[min_i + 1 : min_j + 1][::-1]
        cero = city.index(0)
        if cero !=0:
            city = city[cero:len(city)]+city[0:cero]
        city.remove(0)

    def JobLocalSearch(self,ciudad,trabajo):
        n = len(ciudad)
        suma_ac = [self.TT[0][ciudad[0]]]
        suma = suma_ac[-1]
        makespan = suma_ac[-1]+self.JT[trabajo[0]][ciudad[0]] 
        cmax1 = makespan
        cmax2 = 0
        i1 = 0
        i2 = 0
        i = 0
        while i < n - 1:
            suma += self.TT[ciudad[i]][ciudad[i + 1]]
            suma_ac.append(suma)
            makespan = suma_ac[-1] + self.JT[trabajo[i+1]][ciudad[i+1]]
            if makespan > cmax1:
                cmax2 = cmax1
                cmax1 = makespan
                i2 = i
                i1 = i+1
            elif makespan > cmax2:
                cmax2 = makespan
                i2 = i
            i += 1

        suma += self.TT[ciudad[-1]][0]
        suma_ac.append(suma)
        if suma_ac[-1]>cmax1:
            return

        for i in range(n):
            if i != i1:
                cambio_i1 = suma_ac[i1] + self.JT[trabajo[i]][ciudad[i1]]
                cambio_i = suma_ac[i] + self.JT[trabajo[i1]][ciudad[i]]
                if cambio_i1 < cmax1 and cambio_i < cmax1:
                    trabajo[i1],trabajo[i] = trabajo[i],trabajo[i1]
                    return

    def mutSet(self,city,jobs):
        value2 = random.uniform(0,1)
        MS1 = self.parameters["MS1"]
        P_EM = self.parameters["P_EM"]
        P_RM = self.parameters["P_RM"]
        P_SM = self.parameters["P_SM"]
        P_2OPT = self.parameters["P_2OPT"]
        MS2 = self.parameters["MS2"]
        P_JLS = self.parameters["P_JLS"]
        P_JEM = self.parameters["P_JEM"]

        if value2<MS1:
            value = random.uniform(0, 1)
            if value < P_EM:
                self.ExchangeMutation(city)
            elif value < P_EM+P_RM:
                self.ReverseMutation(city)
            elif value <P_EM+P_RM+P_SM:
                self.SuccessorMutation(city)
            elif value <P_EM+P_RM+P_SM+P_2OPT:
                self.DosOpt(city)

        
        value3 = random.uniform(0,1)
        if value3<MS2: 
            value = random.uniform(0, 1)
            if value < P_JLS:
                self.JobLocalSearch(city,jobs)
            elif value < P_JLS + P_JEM:
                self.ExchangeMutation(jobs)

    def run(self):
        random.seed(self.seed)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, typecode='i', fitness=creator.FitnessMin)
        self.initial_population()
        self.initialize()
        if self.compare == False:
            print(self.log[-1]["gen"],0,"%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"])
        
        else:
            self.df = pd.DataFrame(columns=["gen","avg","min","std"])
            self.df.loc[0] = [self.log[-1]["gen"], "%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"], "%.2f"%self.log[-1]["std"]]
        
        initial_time = time.time()
        while self.g<self.parameters["IT"] and time.time()-initial_time<=self.parameters["TIMELIMIT"]:
            self.g += 1

            offspring = self.toolbox.select(self.pop, int(len(self.pop)-self.elite_size))
            offspring = self.crossover(offspring)
            offspring = self.mutation(offspring)
            
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit < self.best_aux:
                    self.best_aux = fit 
                    self.best_iteration = self.g
            
            self.JobLocalSearch(self.hof.items[0][0],self.hof.items[0][1])
            
            offspring.extend(self.hof.items)
            self.hof.update(offspring)
            self.pop[:] = offspring

            record = self.stats.compile(offspring)
            self.log.record(gen=self.g, evals=len(offspring), **record)
            if self.compare == False:
                print(self.log[-1]["gen"],"%.5f"%(time.time()-initial_time), "%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"])
                self.df = None
            else:
                self.df.loc[self.g] = [self.log[-1]["gen"], "%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"],"%.2f"%self.log[-1]["std"]]
            
            top = tools.selBest(offspring, k=1)
            self.top = top[-1]
        self.total_time = time.time()-initial_time

        if self.compare == False:
            print('Costo  : %.2f' % self.log[-1]["min"])
            print("Tiempo : %f" % self.total_time)

    def get_results(self):
        """
        Return min, total time, avg, dataframe, best iteration, initial pop fitness, best initial, population time
        """
        return "%.2f"%self.log[-1]["min"],self.total_time,"%.2f"%self.log[-1]["avg"],self.df,self.best_iteration,self.initial_pop_fitness,self.best_initial,self.population_time
    
    def print_results(self):
        m,t,p,d,mejor_iteracion,promedio_inicial,mejor_inicial,t_poblacion = self.get_results()
        print("{:<6}{:<8}{:<12}{:<12}{:<10}{:<10}{:<10}{:<10}{:<12}{:<12}".format(self.seed,self.size,self.instance,"%.2f"%float(m),"%.2f"%float(p),"%.3f"%t,"%.2f"%float(mejor_inicial),"%.2f"%float(promedio_inicial),mejor_iteracion,round(t_poblacion,2)))
    
    def get_solution(self):
        return self.top
