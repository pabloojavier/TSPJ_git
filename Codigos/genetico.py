import random
import pandas as pd
import matplotlib.pyplot as plt
import operator
import numpy as np


class Solucion:
    global TT
    global JT
    def __init__(self,lista):
        self.lista = lista
        self.valor = 0
    def fitness(self):

        return self.valor
    def ruta(self):
        return self.lista
    def __str__(self):
        return f"{self.lista}"


def parameters(excel):
    TT = pd.read_excel(excel,sheet_name="TT",index_col=0)
    JT = pd.read_excel(excel,sheet_name="JT",index_col=0)
    #coord = pd.read_excel(excel,sheet_name = "coord",index_col=0,header=0)
    #coord_x = coord["coord_x"]
    #coord_y = coord["coord_y"]
    #print(JT[2][1])
    #JT[NODOS][TRABAJOS]
    n = len(list(TT.index.values))
    nodes = [i for i in range(n)]
    arch = [(i,j) for i in nodes for j in nodes if i !=j]
    travel_time = {(i,j): TT[i][j] for i,j in arch}
    job_time = {(i,j): JT[i][j] for i in nodes for j in nodes}

    return nodes,arch,travel_time,job_time

def solucion_inicial(individuos,nodos,arcos,TT,JT):
    soluciones = []
    i=0
    while len(soluciones)<individuos:
        lista = [0]
        while len(lista)<len(nodos):
            aux = np.random.randint(1,len(nodos))
            if aux not in lista:
                lista.append(aux)
        lista.append(0)
        soluciones.append(Solucion(lista))
    return soluciones


nodos,arcos,TT,JT= parameters("/Users/pablogutierrezaguirre/Desktop/Proyecto profe carlos/TSPJ/TSPJ_gr17.xlsx")
lista = solucion_inicial(5,nodos,arcos,TT,JT)
for i in lista:
    print(i)