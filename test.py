#  Si dichiara che il contenuto di questo file e in ogni sua parte
#  opera originale dell'autore.
#
#  Lorenzo Beretta, 536242, loribere@gmail.com

from cg import *
from scipy.sparse.linalg import cg
import numpy as np
import csv
import matplotlib.pyplot as plt

''' This is an utility file containig function that may be useful while
making experiment automating some tasks.

'''
def run_experiment(file_path, rng, v1, v2):
    edges, b = read_DIMACS(file_path)

    edges = [(e[0], e[1], np.exp(np.random.uniform(-rng, rng))) for e in edges]

    x1, f1, itn1, t1, info1 = solve(edges, b, maxiter=1000)

    x2, f2, itn2, t2, info2 = solve(edges, b, maxiter=1000, algo=cg)


    n = len(b)
    A = make_operator(edges, n) 

    cg1_err_abs = np.linalg.norm(A * x1 - b)
    cg1_err_rel = cg1_err_abs / np.linalg.norm(b)

    cg2_err_abs = np.linalg.norm(A * x2 - b)
    cg2_err_rel = cg2_err_abs / np.linalg.norm(b)

    v1.append((rng, itn1, t1, cg1_err_rel))
    v2.append((rng, itn2, t2, cg2_err_rel))


def run(v1, v2):
    prefix = "mcf_generator/3000/netgen-3000-3-"
    suffix = "-b-b-s.dmx"

    for i in range (1, 6):
        file_path = prefix + i.__str__() + suffix
        for j in range (1, 11):
            for h in range (20):
                run_experiment(file_path, j*10, v1, v2)


def file_print(v, file_name):
    v = [[w.__str__() for w in z ] for z in v]
    with open(file_name, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(v)

    
