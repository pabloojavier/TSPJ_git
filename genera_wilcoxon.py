bls = ["1","2"]
bks = [float(linea) for linea in open("output/wilcoxon_bls/bks.txt")]
MGA = [float(linea) for linea in open("output/wilcoxon_bls/MGA.txt")]
gap_mga = [(MGA[j]-bks[j])/bks[j]*100 for j in range(len(MGA))]

for i in bls:
    best_exp = [float(linea) for linea in open(f"output/wilcoxon_bls/bls{i}.txt")]
    gap_exp = [((best_exp[j]-bks[j])/bks[j]*100) for j in range(len(best_exp))]
    archivo = open(f"output/wilcoxon_bls/gap_wilcoxon_bls{i}.txt","a")
    for j in range(len(best_exp)):
        archivo.write(f"{gap_exp[j]} {gap_mga[j]}\n")
    archivo.close()

