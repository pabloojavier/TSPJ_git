import pandas as pd

df = pd.DataFrame()
for i in ["large","medium","small","tsplib"]:
    for j in ["AG7","AG8"]:
        ruta = f"output/exp_final_real_resumen/ga_{i}_{j}.txt"
        archivo = pd.read_csv(ruta,header=None,delim_whitespace=True)
        archivo["grupo"]=[i for k in range(len(archivo))]
        archivo["exp"] = [j for k in range(len(archivo))]
        df = pd.concat([df,archivo])

df.to_csv("todo.csv",sep=";",decimal = ",")