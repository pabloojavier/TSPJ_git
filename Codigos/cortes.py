import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio


rutas = []
largo = len(rutas)
arcos = []
restricciones = []
contador = 0

def graficar():
    frames = max(len(rutas),len(arcos))
    fig, ax = plt.subplots()
    def update(i):
        ax.clear()
        ax.set_xlim(-2,103)
        ax.set_ylim(-2,103)
        x = [points[j][0] for j in range(len(points))]
        y = [points[j][1] for j in range(len(points))]
        ax.scatter(x, y,c="blue")
        
        subtour_graph = rutas[i]
        for j in range(len(subtour_graph)-1):
            x1 = points[subtour_graph[j]][0]
            x2 = points[subtour_graph[j+1]][0]
            y1 = points[subtour_graph[j]][1]
            y2 = points[subtour_graph[j+1]][1]
            ax.plot([x1,x2],[y1,y2],color = "red",zorder=2)

        x1 = points[subtour_graph[-1]][0]
        x2 = points[subtour_graph[0]][0]
        y1 = points[subtour_graph[-1]][1]
        y2 = points[subtour_graph[0]][1]
        ax.plot([x1,x2],[y1,y2],color = "red",zorder=2)
        
        tour_actual = arcos[i]
        for j in range(len(tour_actual)):
            x1 = points[tour_actual[j][0]][0]
            x2 = points[tour_actual[j][1]][0]
            y1 = points[tour_actual[j][0]][1]
            y2 = points[tour_actual[j][1]][1]
            ax.plot([x1,x2],[y1,y2],color = "black",zorder=1)

        ax.set_title(f"Restricciones añadidas:{restricciones[i]}")

    ani = FuncAnimation(fig, update, frames=frames, interval=500,repeat=False)
    plt.show()


def subtourelim(model, where):
    global contador
    global restricciones
    
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals)
        rutas.append(tour)
        arcos.append([tupla for tupla, valor in vals.items() if valor >0.5])
        #print([(tupla,valor) for tupla, valor in vals.items() if valor >0.5])
        #print(vals)
        #print("a")
        #exit(0)
        if len(tour) < n:
            contador += 1
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour)-1)
            #print("+".join([f"({i},{j})" for i,j in combinations(tour, 2)])+f"<= {len(tour)-1}")
        restricciones.append(contador)


def subtourelim2(modelo, donde):
    global n
    global contador
    if donde == GRB.Callback.MIPSOL:
        valoresX = modelo.cbGetSolution(modelo._vars)
        # encontrar ciclo más pequeño
        tour = [i for i in range(n+1)]
        subtour2(tour, valoresX,n)
        rutas.append(tour)
        arcos.append([tupla for tupla, valor in valoresX.items() if valor >0.5])
        if len(tour) < n:
            contador += 1
            # agregar cortes de elimination de subtour DFJ2
            tour2 = [i for i in range(n) if i not in tour]
            modelo.cbLazy(gp.quicksum(modelo._vars[i, j] for i in tour for j in tour2) >= 1)
        restricciones.append(contador)

def subtour2(subruta, vals,n):
    # obtener una lista con los arcos parte de la solucións
    arcos = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    noVisitados = list(range(n))
    while noVisitados: # true if list is non-empty
        ciclo = []
        vecinos = noVisitados
        while vecinos:
            actual = vecinos[0]
            ciclo.append(actual)
            noVisitados.remove(actual)
            vecinos = [j for i, j in arcos.select(actual, '*') if j in noVisitados]
        if len(subruta) > len(ciclo):
            subruta[:] = ciclo

def subtour(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys()
                         if vals[i, j] > 0.5)
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle



n = 50
random.seed(203696760)
points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]

dist = {(i, j):
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)
    env.start()
    with gp.Model(env=env) as m:
        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
        for i, j in vars.keys():
            vars[j, i] = vars[i, j]

        m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))

        # Optimize model
        m._vars = vars
        m.Params.LazyConstraints = 1
        m.optimize(subtourelim)
        vals = m.getAttr('X', vars)
        tour = subtour(vals)
        arcos.append([tupla for tupla, valor in vals.items() if valor >0.5])
        rutas.append(tour)
        restricciones.append(contador)
        graficar()
        print('')
        print('Optimal cost: %g' % m.ObjVal)
        print('')