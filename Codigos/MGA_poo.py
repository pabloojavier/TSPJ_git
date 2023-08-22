from Source.MGA import MGA

mga = MGA("large",1,seed = 0)
mga.compare = False
mga.run()
#mga.print_results()
print(mga.get_solution())
