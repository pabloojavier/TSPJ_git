exp = [str(i) for i in range(1,7)]
bks = [float(linea) for linea in open("output/wilcoxon_exp/bks.txt")]
MGA = [float(linea) for linea in open("output/MGA_exp_todo.txt")]
gap_mga = [(MGA[j]-bks[j])/bks[j]*100 for j in range(len(MGA))]

for i in exp:
    best_exp = [float(linea) for linea in open(f"output/wilcoxon_exp/wilcoxon_exp{i}.txt")]
    gap_exp = [((best_exp[j]-bks[j])/bks[j]*100) for j in range(len(best_exp))]
    archivo = open(f"output/wilcoxon_exp/gap_wilcoxon_exp{i}.txt","a")
    for j in range(len(best_exp)):
        archivo.write(f"{gap_exp[j]} {gap_mga[j]}\n")
    archivo.close()

