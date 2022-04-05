import os

list(set([x.split('.')[0] for x in os.listdir('comsol_results/1dconstantslots')]))
