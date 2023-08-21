#!/usr/bin/env python3.9
from operator import index
from numpy.random.mtrand import rand, seed
import matplotlib.pyplot as plt
import random
import time
#import animated_visualizer as animacion
import pandas as pd
import os
import array
import sys
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from itertools import combinations

import gurobipy as gp
from gurobipy import *

import requests
import tsplib95
import lkh
from typing import Dict, List, Callable, Any
from Source.Problem import Problem


path = "Codigos/"
flagRoute = False
gurobi_tour = []

class MGA(Problem):
    def __init__(self,
                 size,
                 instance,
                 parameters,
                 compare):
        super().__init__(size,instance)
        self.parameters = parameters
        self.g = 0
        self.best_iteration = 0 
        self.compare = compare

    def initial_population(self):
        self.toolbox = base.Toolbox()
        t_aux = time.time()
        self.toolbox.register("cromo1",self.generarRuta,self.n-1)
        self.toolbox.register("cromo2", self.creador_trabajos, self.n)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,(self.toolbox.cromo1, self.toolbox.cromo2), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.pop = self.toolbox.population(n=self.parameters["POBLACION"])
        for i in self.pop:
            i[1] = self.crear_trabajos(i[0],i[1]) 
        self.fitnesses = list(map(self.toolbox.evaluate, self.pop))
        self.initial_pop_fitness = numpy.mean(self.fitnesses)
        self.best_initial = numpy.min(self.fitnesses)
        self.best_aux = self.best_initial

        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit
        self.population_time = time.time()-t_aux

    def initialize(self):
        self.toolbox.register("mate",self.cruzamiento)
        self.toolbox.register("mutate", self.mutSet)
        self.toolbox.register("select", tools.selTournament, tournsize=self.parameters["TOURN"])
        self.toolbox.register("evaluate", self.fitness_functions)
        self.elite_size = self.parameters["POBLACION"]*self.parameters["ELITE"]
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

    def overall_algorithm(self):
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

            #Selection
            offspring = self.toolbox.select(self.pop, int(len(self.pop)-self.elite_size))

            # Crossover
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.parameters["CXPB"]:
                    self.toolbox.mate(child1[0], child2[0])
                    self.toolbox.mate(child1[1], child2[1])
                    del child1.fitness.values
                    del child2.fitness.values
                
            # Mutation
            for mutant in offspring:
                if random.random() < self.parameters["MUTPB"]:
                    self.toolbox.mutate(mutant[0],mutant[1])
                    del mutant.fitness.values       
            
            # Evaluación
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit < self.best_aux:
                    self.best_aux = fit 
                    self.best_iteration = self.g

            self.LSJA(self.hof.items[0][0],self.hof.items[0][1])

            # ELITISM
            offspring.extend(self.hof.items)
            self.hof.update(offspring)
            self.pop[:] = offspring

            record = self.stats.compile(offspring)
            self.log.record(gen=self.g, evals=len(offspring), **record)
            if self.compare == False:
                print(self.log[-1]["gen"],"%.5f"%(time.time()-initial_time), "%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"])
            else:
                self.df.loc[self.g] = [self.log[-1]["gen"], "%.2f"%self.log[-1]["avg"], "%.2f"%self.log[-1]["min"],"%.2f"%self.log[-1]["std"]]
        
        self.total_time = time.time()-initial_time
        minimo, promedio = self.log.select("min", "avg")

        if self.compare == False:
            print('Costo  : %.2f' % self.log[-1]["min"])
            print("Tiempo : %f" % self.total_time)
            return None,None,None,None,None,None,None,None

        else:
            return "%.2f"%self.log[-1]["min"],self.total_time,"%.2f"%self.log[-1]["avg"],self.df,self.best_iteration,self.initial_pop_fitness,self.best_initial,self.population_time

    
        

parameters = {"POPULATION":100,
              "ELITE":0.2}
mga = MGA("tsplib","gr17",parameters)
print(mga.n)

