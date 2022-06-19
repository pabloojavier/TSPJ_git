from ast import arg
from code import InteractiveConsole
from termcolor import colored
import pandas as pd
import numpy as np
from tabulate import tabulate
import sys

def redondear(numero,tipo1):
    return "%.3f"%numero if tipo1 != "Latex" else "%.3f"%numero

def minimo(a):
    aux = 1000
    for i in range(len(a)):
        if a[i]<aux:
            aux = a[i]
    return aux

def colorTABLA2(numero,tipo,it):
    if np.amin(np.array([float(gapI[it][:-2]),float(gapII[it][:-2]),float(gapIII[it][:-2]),float(gapIV[it][:-2])])) >= numero:
        return "\color[HTML]{32CB00}" + "%.3f"%numero if tipo=="Latex" else colored("%.3f"%numero,"green")
    elif np.amax(np.array([float(gapI[it][:-2]),float(gapII[it][:-2]),float(gapIII[it][:-2]),float(gapIV[it][:-2])])) <= numero:
        return "\color[HTML]{FE0000}" + "%.3f"%numero if tipo=="Latex" else colored("%.3f"%numero,"red")

    elif np.amin(np.array([float(gapI[it][:-2]),float(gapII[it][:-2]),float(gapIII[it][:-2]),float(gapIV[it][:-2])])) <= numero:
        return "%.3f"%numero if tipo=="Latex" else "%.3f"%numero

    elif np.amax(np.array([float(gapI[it][:-2]),float(gapII[it][:-2]),float(gapIII[it][:-2]),float(gapIV[it][:-2])])) >= numero:
        return "%.3f"%numero if tipo=="Latex" else "%.3f"%numero
        
def promedio(lista,decimales = 2):
    a = sum(lista)/len(lista)
    if decimales ==2:
        return float("%.2f"%a)
    else:
        return float("%.3f"%a)

def gap(lista1,referencia):
    nueva = []
    for u in range((len(referencia))):
        valor = ((1-lista1[u]/referencia[u])*(-100))
        if valor == 0:
            valor = 0
        nueva.append(valor)
    return nueva

def output_to_list(ruta):
    archivo = open(ruta,"r")
    lineas=[linea.split() for linea in archivo]
    instancias = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
    minimos = [min( [float(lineas[i][2]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    avg = [promedio([float(lineas[i][2]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    tiempo = [promedio([float(lineas[i][4]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    mejor_inicial = [min( [float(lineas[i][5]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    avg_mejor_inicial = [promedio([float(lineas[i][6]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    iteracion = [promedio([float(lineas[i][7]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    t_pob = [promedio([float(lineas[i][8]) for i in range(len(lineas)) if lineas[i][1]==j]) for j in instancias]
    archivo.close()


    #print([lineas[i][2] for i in range(len(lineas)) for j in instancias if lineas[i][1]==j ])
    return minimos,avg,tiempo,mejor_inicial,avg_mejor_inicial,iteracion,t_pob

def agVsMosayebiV2(tiempo,min,avg,tipo):
    instancias = ["","gr17-J","gr21-J","gr24-J","fri26-J","bays29-J","gr48-J","eil51-J","berlin52-J","eil76-J","eil101-J","promedio"]

    I =  ['I', 2760.0, 7956.0, 1818.0, 1326.0, 2978.0, 7692.0, 648.71, 11230.49, 847.99, 975.94, 3823.31]
    II =  ['II', 2991.0, 8428.0, 1927.0, 1326.0, 2943.0, 7499.0, 651.0, 11362.0, 845.0, 1003.0, 3897.5]
    III =  ['III', 2760.0, 7962.0, 1818.0, 1326.0, 2940.0, 7692.0, 648.71, 11225.77, 847.65, 975.94, 3819.61]
    IV =  ['IV', 2760.0, 7962.0, 1853.0, 1326.0, 2962.0, 7630.0, 640.2, 11427.66, 822.46, 998.92, 3838.22]
    gams =  ['gams', 2760.0, 7788.0, 1806.0, 1283.0, 2937.0, 7288.0, 630.0, 11087, 802.0, 947.4, 3732.84]

    tI =  ['I', 0.13, 0.21, 0.36, 0.22, 0.29, 0.89, 3.86, 0.96, 1.68, 14.75, 2.33]
    tII =  ['II', 0.13, 0.23, 0.21, 0.27, 0.35, 2.48, 3.3, 1.05, 2.61, 13.83, 2.45]
    tIII =  ['III', 0.19, 0.27, 0.47, 0.35, 0.57, 1.36, 5.9, 1.41, 2.05, 22.52, 3.51]
    tIV =  ['IV', 0.15, 0.21, 0.34, 0.41, 0.6, 1.23, 5.69, 2.3, 3.32, 28.37, 4.26]  

    global gapI
    global gapII
    global gapIII
    global gapIV
    gapI   = ["I"]    +  [redondear((gap(I[1:]    , gams[1:]) + [promedio(gap(I[1:]   , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
    gapII  = ["II"]   +  [redondear((gap(II[1:]   , gams[1:]) + [promedio(gap(II[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
    gapIII = ["III"]  +  [redondear((gap(III[1:]  , gams[1:]) + [promedio(gap(III[1:] , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
    gapIV  = ["IV"]   +  [redondear((gap(IV[1:]   , gams[1:]) + [promedio(gap(IV[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
    gapmin = ["min"]  +  [colorTABLA2((gap(min+[promedio(min)],gams[1:]) + [promedio(gap(min+[promedio(min)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]
    gapavg = ["avg"]  +  [colorTABLA2((gap(avg+[promedio(avg)],gams[1:]) + [promedio(gap(avg+[promedio(avg)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]

    min = list(map(lambda x: float("%.1f"%x),min))
    avg = list(map(lambda x: float("%.1f"%x),avg))
    tiempo = list(map(lambda x: float("%.1f"%x),tiempo))

    min.append(promedio(min))
    avg.append(promedio(avg))
    tiempo.append(promedio(tiempo))

    min.insert(0,"AGM")
    avg.insert(0,"AGA")
    tiempo.insert(0,"AG")

    barras = [""]*len(gams) if tipo!="Latex" else ["$\\vert$"]*len(gams)

    if tipo =="Latex":
        columnas = ["name","_","_","_","CMAX","_","_","_","_","_","_","gap \%","_","_","_","_","_","","sec","_","_"]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],gapI[i],gapII[i],gapIII[i],gapIV[i],gapmin[i],gapavg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]
        print(df.to_latex(index=False,escape=False))

    else:
        columnas = ["name","","","","CMAX","","","","","","","gap %","","","","","","","sec","",""]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],gapI[i],gapII[i],gapIII[i],gapIV[i],gapmin[i],gapavg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]

        print(tabulate(df, headers='keys', tablefmt='psql',showindex=False))

def agVsMosayebiV1(tiempo,min,avg,tipo):
    instancias = ["","gr17-J","gr21-J","gr24-J","fri26-J","bays29-J","gr48-J","eil51-J","berlin52-J","eil76-J","eil101-J","promedio"]

    I =  ["cmax", 2760.0, 7956.0, 1818.0, 1326.0, 2978.0, 7692.0, 648.71, 11230.49, 847.99, 975.94, 3823.31]
    II =  ['cmax', 2991.0, 8428.0, 1927.0, 1326.0, 2943.0, 7499.0, 651.0, 11362.0, 845.0, 1003.0, 3897.5]
    III =  ['cmax', 2760.0, 7962.0, 1818.0, 1326.0, 2940.0, 7692.0, 648.71, 11225.77, 847.65, 975.94, 3819.61]
    IV =  ['cmax', 2760.0, 7962.0, 1853.0, 1326.0, 2962.0, 7630.0, 640.2, 11427.66, 822.46, 998.92, 3838.22]
    gams =  ['cmax', 2760.0, 7788.0, 1806.0, 1283.0, 2937.0, 7288.0, 630.0, 11087, 802.0, 947.4, 3732.84]

    tI =  ['I', 0.13, 0.21, 0.36, 0.22, 0.29, 0.89, 3.86, 0.96, 1.68, 14.75, 2.33]
    tII =  ['II', 0.13, 0.23, 0.21, 0.27, 0.35, 2.48, 3.3, 1.05, 2.61, 13.83, 2.45]
    tIII =  ['III', 0.19, 0.27, 0.47, 0.35, 0.57, 1.36, 5.9, 1.41, 2.05, 22.52, 3.51]
    tIV =  ['IV', 0.15, 0.21, 0.34, 0.41, 0.6, 1.23, 5.69, 2.3, 3.32, 28.37, 4.26]  

    global gapI
    global gapII
    global gapIII
    global gapIV
    if tipo != "Latex":
        gapI   = ["gap "]    +  [redondear((gap(I[1:]    , gams[1:]) + [promedio(gap(I[1:]   , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapII  = ["gap "]   +  [redondear((gap(II[1:]   , gams[1:]) + [promedio(gap(II[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapIII = ["gap "]  +  [redondear((gap(III[1:]  , gams[1:]) + [promedio(gap(III[1:] , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapIV  = ["gap "]   +  [redondear((gap(IV[1:]   , gams[1:]) + [promedio(gap(IV[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapmin = ["gap min"]  +  [colorTABLA2((gap(min+[promedio(min)],gams[1:]) + [promedio(gap(min+[promedio(min)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]
        gapavg = ["gap avg"]  +  [colorTABLA2((gap(avg+[promedio(avg)],gams[1:]) + [promedio(gap(avg+[promedio(avg)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]
    else:
        gapI   = ["gap\% "]    +  [redondear((gap(I[1:]    , gams[1:]) + [promedio(gap(I[1:]   , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapII  = ["gap\% "]   +  [redondear((gap(II[1:]   , gams[1:]) + [promedio(gap(II[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapIII = ["gap\% "]  +  [redondear((gap(III[1:]  , gams[1:]) + [promedio(gap(III[1:] , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapIV  = ["gap\% "]   +  [redondear((gap(IV[1:]   , gams[1:]) + [promedio(gap(IV[1:]  , gams[1:]))])[i] , tipo) for i in range(len(gams)-1)]
        gapmin = ["gap min\%"]  +  [colorTABLA2((gap(min+[promedio(min)],gams[1:]) + [promedio(gap(min+[promedio(min)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]
        gapavg = ["gap avg\%"]  +  [colorTABLA2((gap(avg+[promedio(avg)],gams[1:]) + [promedio(gap(avg+[promedio(avg)],gams[1:]))])[i] , tipo,i+1) for i in range(len(gams)-1)]

    min = list(map(lambda x: float("%.1f"%x),min))
    avg = list(map(lambda x: float("%.1f"%x),avg))
    tiempo = list(map(lambda x: float("%.1f"%x),tiempo))

    min.append(promedio(min))
    avg.append(promedio(avg))
    tiempo.append(promedio(tiempo))

    min.insert(0,"min")
    avg.insert(0,"avg")
    tiempo.insert(0,"AG")

    barras = [""]*len(gams) if tipo!="Latex" else ["$\\vert$"]*len(gams)
    if tipo =="Latex":
        columnas = ["name","gams","_","I","_","_","II","_","_","III","_","_","IV","_","_","_","AG","_","_","_","_","_","sec","_","_"]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],barras[i],I[i],gapI[i],barras[i],II[i],gapII[i],barras[i],III[i],gapIII[i],barras[i],IV[i],gapIV[i],barras[i],min[i],avg[i],gapmin[i],gapavg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]
        print(df.to_latex(index=False,escape=False))

    else:
        columnas = ["name","gams","","I","","","II","","","III","","","IV","","","","AG","","","","","","sec","",""]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],barras[i],I[i],gapI[i],barras[i],II[i],gapII[i],barras[i],III[i],gapIII[i],barras[i],IV[i],gapIV[i],barras[i],min[i],avg[i],gapmin[i],gapavg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]

        print(tabulate(df, headers='keys', tablefmt='psql',showindex=False))

def instancias_prueba(ruta):
    #archivo = open("/Users/pablogutierrezaguirre/Desktop/Proyecto profe carlos/pruebas/"+size+".txt","r")
    archivo = open(ruta,"r")
    lineas=[linea.split() for linea in archivo]
    lineas = lineas[3:]
    tiempo_s = [float(lineas[i][6]) for i in range(len(lineas))]
    minimo_s = []
    for i in range(len(lineas)):
        min_abs = float(lineas[i][4])
        minimo_s.append(float("%.2f"%(min_abs)))
    
    print("Promedio: ",float("%.2f"%(sum(minimo_s)/len(minimo_s))))
    print("Tiempo: ",float("%.2f"%(sum(tiempo_s)/len(tiempo_s))))

def agVsMosayebiV2_SINGAP(tiempo,min,avg,tipo):
    instancias = ["","gr17-J","gr21-J","gr24-J","fri26-J","bays29-J","gr48-J","eil51-J","berlin52-J","eil76-J","eil101-J","promedio"]

    I =  ['I', 2760.0, 7956.0, 1818.0, 1326.0, 2978.0, 7692.0, 648.71, 11230.49, 847.99, 975.94, 3823.31]
    II =  ['II', 2991.0, 8428.0, 1927.0, 1326.0, 2943.0, 7499.0, 651.0, 11362.0, 845.0, 1003.0, 3897.5]
    III =  ['III', 2760.0, 7962.0, 1818.0, 1326.0, 2940.0, 7692.0, 648.71, 11225.77, 847.65, 975.94, 3819.61]
    IV =  ['IV', 2760.0, 7962.0, 1853.0, 1326.0, 2962.0, 7630.0, 640.2, 11427.66, 822.46, 998.92, 3838.22]
    gams =  ['gams', 2760.0, 7788.0, 1806.0, 1283.0, 2937.0, 7288.0, 630.0, 11087, 802.0, 947.4, 3732.84]

    tI =  ['I', 0.13, 0.21, 0.36, 0.22, 0.29, 0.89, 3.86, 0.96, 1.68, 14.75, 2.33]
    tII =  ['II', 0.13, 0.23, 0.21, 0.27, 0.35, 2.48, 3.3, 1.05, 2.61, 13.83, 2.45]
    tIII =  ['III', 0.19, 0.27, 0.47, 0.35, 0.57, 1.36, 5.9, 1.41, 2.05, 22.52, 3.51]
    tIV =  ['IV', 0.15, 0.21, 0.34, 0.41, 0.6, 1.23, 5.69, 2.3, 3.32, 28.37, 4.26]  

    min = list(map(lambda x: float("%.1f"%x),min))
    avg = list(map(lambda x: float("%.1f"%x),avg))
    tiempo = list(map(lambda x: float("%.1f"%x),tiempo))

    min.append(promedio(min))
    avg.append(promedio(avg))
    tiempo.append(promedio(tiempo))

    min.insert(0,"AGM")
    avg.insert(0,"AGA")
    tiempo.insert(0,"AG")

    barras = [""]*len(gams) if tipo!="Latex" else ["$\\vert$"]*len(gams)

    if tipo =="Latex":
        columnas = ["name","_","_","_","CMAX","_","_","_","_","_","","sec","_","_"]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]
        print(df.to_latex(index=False,escape=False))

    else:
        columnas = ["name","","","","CMAX","","","","","","","sec","",""]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i]]

        print(tabulate(df, headers='keys', tablefmt='psql',showindex=False))

def agVsMosayebiV2_CMAX(tiempo,min,avg,tipo):
    instancias = ["","gr17-J","gr21-J","gr24-J","fri26-J","bays29-J","gr48-J","eil51-J","berlin52-J","eil76-J","eil101-J","promedio"]

    I =  ['I', 2760.0, 7956.0, 1818.0, 1326.0, 2978.0, 7692.0, 648.71, 11230.49, 847.99, 975.94, 3823.31]
    II =  ['II', 2991.0, 8428.0, 1927.0, 1326.0, 2943.0, 7499.0, 651.0, 11362.0, 845.0, 1003.0, 3897.5]
    III =  ['III', 2760.0, 7962.0, 1818.0, 1326.0, 2940.0, 7692.0, 648.71, 11225.77, 847.65, 975.94, 3819.61]
    IV =  ['IV', 2760.0, 7962.0, 1853.0, 1326.0, 2962.0, 7630.0, 640.2, 11427.66, 822.46, 998.92, 3838.22]
    gams =  ['gams', 2760.0, 7788.0, 1806.0, 1283.0, 2937.0, 7288.0, 630.0, 11087, 802.0, 947.4, 3732.84]

    tI =  ['I', 0.13, 0.21, 0.36, 0.22, 0.29, 0.89, 3.86, 0.96, 1.68, 14.75, 2.33]
    tII =  ['II', 0.13, 0.23, 0.21, 0.27, 0.35, 2.48, 3.3, 1.05, 2.61, 13.83, 2.45]
    tIII =  ['III', 0.19, 0.27, 0.47, 0.35, 0.57, 1.36, 5.9, 1.41, 2.05, 22.52, 3.51]
    tIV =  ['IV', 0.15, 0.21, 0.34, 0.41, 0.6, 1.23, 5.69, 2.3, 3.32, 28.37, 4.26]  

    min = list(map(lambda x: float("%.1f"%x),min))
    avg = list(map(lambda x: float("%.1f"%x),avg))
    tiempo = list(map(lambda x: float("%.1f"%x),tiempo))

    min.append(promedio(min))
    avg.append(promedio(avg))
    tiempo.append(promedio(tiempo))

    min.insert(0,"AGM")
    avg.insert(0,"AGA")
    tiempo.insert(0,"AG")

    barras = [""]*len(gams) if tipo!="Latex" else ["$\\vert$"]*len(gams)

    if tipo =="Latex":
        columnas = ["name","_","_","_","CMAX","_","_","_"]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i]]
        print(df.to_latex(index=False,escape=False))

    else:
        columnas = ["name","","","","CMAX","","",""]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i]]

        print(tabulate(df, headers='keys', tablefmt='psql',showindex=False))

def agVsMosayebiV2_stats(tiempo,min,avg,mejor_inicial,avg_mejor_inicial,iteracion,t_pob,tipo):
    instancias = ["","gr17-J","gr21-J","gr24-J","fri26-J","bays29-J","gr48-J","eil51-J","berlin52-J","eil76-J","eil101-J","promedio"]

    I =  ['I', 2760.0, 7956.0, 1818.0, 1326.0, 2978.0, 7692.0, 648.71, 11230.49, 847.99, 975.94, 3823.31]
    II =  ['II', 2991.0, 8428.0, 1927.0, 1326.0, 2943.0, 7499.0, 651.0, 11362.0, 845.0, 1003.0, 3897.5]
    III =  ['III', 2760.0, 7962.0, 1818.0, 1326.0, 2940.0, 7692.0, 648.71, 11225.77, 847.65, 975.94, 3819.61]
    IV =  ['IV', 2760.0, 7962.0, 1853.0, 1326.0, 2962.0, 7630.0, 640.2, 11427.66, 822.46, 998.92, 3838.22]
    gams =  ['gams', 2760.0, 7788.0, 1806.0, 1283.0, 2937.0, 7288.0, 630.0, 11087, 802.0, 947.4, 3732.84]

    tI =  ['I', 0.13, 0.21, 0.36, 0.22, 0.29, 0.89, 3.86, 0.96, 1.68, 14.75, 2.33]
    tII =  ['II', 0.13, 0.23, 0.21, 0.27, 0.35, 2.48, 3.3, 1.05, 2.61, 13.83, 2.45]
    tIII =  ['III', 0.19, 0.27, 0.47, 0.35, 0.57, 1.36, 5.9, 1.41, 2.05, 22.52, 3.51]
    tIV =  ['IV', 0.15, 0.21, 0.34, 0.41, 0.6, 1.23, 5.69, 2.3, 3.32, 28.37, 4.26]  

    min = list(map(lambda x: float("%.1f"%x),min))
    avg = list(map(lambda x: float("%.1f"%x),avg))
    tiempo = list(map(lambda x: float("%.1f"%x),tiempo))

    min.append(promedio(min))
    avg.append(promedio(avg))
    tiempo.append(promedio(tiempo))

    min.insert(0,"AGM")
    avg.insert(0,"AGA")
    tiempo.insert(0,"AG")

    mejor_inicial = list(map(lambda x: float("%.1f"%x),mejor_inicial))
    avg_mejor_inicial = list(map(lambda x: float("%.1f"%x),avg_mejor_inicial))
    iteracion = list(map(lambda x: float("%.1f"%x),iteracion))
    t_pob = list(map(lambda x: float("%.1f"%x),t_pob))

    mejor_inicial.append(promedio(mejor_inicial))
    avg_mejor_inicial.append(promedio(avg_mejor_inicial))
    iteracion.append(promedio(iteracion))
    t_pob.append(promedio(t_pob))

    mejor_inicial.insert(0,"mejor_i")
    avg_mejor_inicial.insert(0,"avg mejor_i")
    iteracion.insert(0,"IT")
    t_pob.insert(0,"t_pob")




    barras = [""]*len(gams) if tipo!="Latex" else ["$\\vert$"]*len(gams)

    if tipo =="Latex":
        columnas = ["name","_","_","_","CMAX","_","_","_","_","_","","sec","_","_","_","_","Stats","_","_"]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i],barras[i],mejor_inicial[i],avg_mejor_inicial[i],iteracion[i],t_pob[i]]
        print(df.to_latex(index=False,escape=False))

    else:
        columnas = ["name","","","","CMAX","","","","","","","sec","","","","","Stats","",""]
        df = pd.DataFrame(columns=columnas)
        for i in range(len(instancias)):
            #print(instancias[i],mejor_inicial[i])
            try:
                df.loc[i] = [instancias[i],gams[i],I[i],II[i],III[i],IV[i],min[i],avg[i],barras[i],tI[i],tII[i],tIII[i],tIV[i],tiempo[i],barras[i],mejor_inicial[i],avg_mejor_inicial[i],iteracion[i],t_pob[i]]
            except:
                pass
        print(tabulate(df, headers='keys', tablefmt='psql',showindex=False))

def instancia_prueba_stats(ruta):
    #archivo = open("/Users/pablogutierrezaguirre/Desktop/Proyecto profe carlos/pruebas/"+size+".txt","r")
    archivo = open(ruta,"r")
    lineas=[linea.split() for linea in archivo]
    instancias = list(map(lambda x:str(x),sorted(list(set([int(lineas[i][3]) for i in range(len(lineas))])))))


    minimos = [min( [float(lineas[i][4]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    avg = [promedio([float(lineas[i][4]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    tiempos = [promedio([float(lineas[i][6]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    mejor_i =  [min( [float(lineas[i][7]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    avg_mejor_i = [promedio([float(lineas[i][8]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    iteracion = [promedio([float(lineas[i][9]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]
    t_pob = [promedio([float(lineas[i][10]) for i in range(len(lineas)) if lineas[i][3]==j]) for j in instancias]


    print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}".format("ins","min","avg","time","mejor_in","avgmejor_in","it_avg","t_pob"))
    for i in range(len(instancias)):
        print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<15}{:<10}{:<10}".format(instancias[i],minimos[i],avg[i],tiempos[i],mejor_i[i],avg_mejor_i[i],iteracion[i],t_pob[i]))
    

ruta ="output/tsplib/3.txt"

#Para instancias TSPLIB
# minimos,avgs,tiempos,mejor_inicial,avg_mejor_inicial,iteracion,t_pob = output_to_list(ruta)
# agVsMosayebiV2_stats(tiempos,minimos,avgs,mejor_inicial,avg_mejor_inicial,iteracion,t_pob,"La1tex")
# agVsMosayebiV2_SINGAP(tiempos,minimos,avgs,"Late1x")

#Para instancias SMALL, MEDIUM Y LARGE
#instancia_prueba_stats("output/small/small1.txt")
instancias_prueba("output/large/cluster.txt")
#instancias_prueba("/Users/pablogutierrezaguirre/Desktop/Proyecto profe carlos/Codigos/nuevas versiones/small_ga_04.txt")