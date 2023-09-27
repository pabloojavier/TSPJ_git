from Source.MGA import MGA

mga = MGA("tsplib","gr17",seed = 0)
mga.parameters["IT"] = 500
mga.compare = True
mga.run()
mga.print_results()
print(mga.get_solution())
print(mga.fitness_functions(mga.get_solution()))
