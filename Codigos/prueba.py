import os 
path = "Codigos/"
def obtener_cortes():
    archivo = open(f"{path}logfile.txt","r")
    lineas = [linea.split("\n") for linea in archivo]
    archivo.close()
    os.remove(f"{path}logfile.txt")
    cortes = {}
    aux = []
    flag = 0
    contador = 0
    for linea in lineas[::-1]:
        if "Explored" in linea[0]:
            flag = 1
            continue
        
        if flag == 1:
            aux.append(linea)
            contador +=1
        
        if contador >25:
            return 0

        if "Cutting planes:" in linea:
            break
    
    del linea
    aux.pop(0)
    aux.pop(-1)
    total = 0
    for i in range(len(aux)):
        nombre = aux[i][0].split(":")[0][2:]
        cantidad = int(aux[i][0].split(":")[1])
        total += cantidad
        cortes[nombre] = cantidad
    del aux
    return (cortes,total)

    

obtener_cortes()