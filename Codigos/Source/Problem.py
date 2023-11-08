import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import tsplib95
import lkh
import sys
import warnings
import networkx as nx
from typing import Dict, List, Callable, Any,Tuple
warnings.filterwarnings("ignore")

if os.getcwd().split("/")[-1] != "Codigos":
    path = "Codigos/"
else:
    path = ""

class Problem:
    def __init__(self,size,instance):
        self.size :str = size.lower()
        self.instance : str = instance
        self.batch = (int(self.instance)-1)//25+1 if self.size in ["small","medium","large"] else ""
        
        self.path = path
        self.__parameters()
        self.__solve_lkh()
        
    def __parameters(self):
        if self.size == "tsplib":
            location = self.path+"Data/Tsplib_problems/"
            self.TT = pd.read_csv(location+"TT_"+self.instance+".csv",index_col= None, header = None).fillna(0).to_numpy()
            self.JT = pd.read_csv(location+"JT_"+self.instance+".csv",index_col= None, header = None).fillna(0).to_numpy()

        elif self.size in ("small","medium","large"):
            location = self.path+"Data/"+str(self.size.capitalize())+"_problems/Batch_0"+str(self.batch)+"/TSPJ_"+str(self.instance)+self.size.capitalize()[0]
            self.TT = pd.read_csv(location+"_cost_table_by_coordinates.csv",index_col= None, header = None).fillna(0).to_numpy()
            self.JT = pd.read_csv(location+"_tasktime_table.csv"           ,index_col= None, header = None).fillna(0).to_numpy()

        else:
            warnings.warn("Size problem has not been specified, using test problem")
            self.TT = pd.read_csv(f"{self.path}Data/test/1_TT_paper.csv",index_col= None, header = None).fillna(0).to_numpy()
            self.JT = pd.read_csv(f"{self.path}Data/test/1_JT_paper.csv",index_col= None, header = None).fillna(0).to_numpy()

        self.n = len(self.TT)
        self.cities = [i for i in range(self.n)]
        self.arch = [(i,j) for i in self.cities for j in self.cities if i !=j]

    def __transform_txt(self,location):
        data = open(location,"r")
        lines = [linea.split() for linea in data]
        data.close()
        lines[0][0] = lines[0][0].replace(","," ")
        lines[0][0] = str(10000000)+" "+ lines[0][0][1:]
        for line in lines[1:-1]:
            line[0] = line[0].replace(",,"," 10000000 ")
            line[0] = line[0].replace(","," ")
        lines[-1][0] = lines[-1][0].replace(","," ")
        lines[-1][0] = lines[-1][0][:-1] +" "+str(10000000)
        
        problem_str = ""
        problem_str = "NAME: prueba"+str(self.n)+"\n"
        problem_str += "TYPE: TSP\n"
        problem_str += f"COMMENT: {self.n} cities in Bavaria, street distances (Groetschel,Juenger,Reinelt)\n"
        problem_str += f"DIMENSION: {self.n}\n"
        problem_str += f"EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nDISPLAY_DATA_TYPE: TWOD_DISPLAY\nEDGE_WEIGHT_SECTION\n"
        for i in lines:
            problem_str += i[0]+"\n"
        return problem_str


    def __solve_lkh(self):
        if self.size in ("small","medium","large"):
            instance_location = self.path+"Data/"+str(self.size.capitalize())+"_problems/Batch_0"+str(self.batch)+"/TSPJ_"+str(self.instance)+self.size.capitalize()[0]+"_cost_table_by_coordinates.csv"
        elif self.size == "tsplib":
            instance_location = self.path+f"Data/Tsplib_problems/TT_{self.instance}.csv"
        else:
            self.lkh_route = [i for i in range(self.n)]
            return

        problem = tsplib95.parse(self.__transform_txt(instance_location))
        solver_path = self.path+'LKH-3.0.7/LKH'

        ciudad = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)[0]
        self.lkh_route =  [i-1 for i in ciudad]

    def get_lkh_route(self):
        return self.lkh_route
    
    def fitness_functions(self,solution:Tuple[List,List]):
        """
        Get the objective function of *solution*
        """
        ciudad = solution[0]
        trabajo = solution[1]
        n = len(ciudad)
        suma_ac = [self.TT[0][ciudad[0]]]
        suma = suma_ac[-1]
        maxtime = suma_ac[-1]+self.JT[ciudad[0]][trabajo[0]]
        cmax = 0
        i = 0
        while i < n -1:
            suma += self.TT[ciudad[i]][ciudad[i + 1]]
            suma_ac.append(suma)
            maxtime = suma_ac[-1] + self.JT[ciudad[i+1]][trabajo[i+1]]
            if maxtime > cmax:
                cmax = maxtime
            i += 1
        suma += self.TT[ciudad[-1]][0]
        suma_ac.append(suma)
        if suma_ac[-1]>cmax:
            cmax = suma_ac[-1]
        return cmax,

