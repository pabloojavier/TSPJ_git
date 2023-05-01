import pandas as pd

df = pd.DataFrame()
for i in ["large","medium","small","tsplib"]:
    for j in ["AG1","AG2","AG3","AG4","AG5","AG6","TL_MGA"]:
        ruta = f"output/exp_final_real_resumen/ga_{i}_{j}.txt"
        archivo = pd.read_csv(ruta,header=None,delim_whitespace=True)
        archivo["grupo"]=[i for k in range(len(archivo))]
        archivo["exp"] = [j for k in range(len(archivo))]
        df = pd.concat([df,archivo])

df.to_csv("todo.csv",sep=";",decimal = ",")