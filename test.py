from cg import *
from scipy.sparse.linalg import cg
import numpy as np


def run_experiment(file_path, v1, v2, rng):
    edges, b = read_DIMACS(file_path)

    edges = [(e[0], e[1], np.exp(np.random.uniform(-rng, rng))) for e in edges]

    x1, f1, itn1, t1, info1 = solve(edges, b, maxiter=1000)

    x2, f2, itn2, t2, info2 = solve(edges, b, maxiter=1000, algo=cg)

    x_dev_abs = np.linalg.norm(x1 - x2)
    x_dev_rel = x_dev_abs / np.linalg.norm(x2)

    f_dev_abs = np.linalg.norm(f1 - f2)
    f_dev_rel = f_dev_abs / np.linalg.norm(f2)

    n = len(b)
    A = make_operator(edges, n) 

    cg1_err_abs = np.linalg.norm(A * x1 - b)
    cg1_err_rel = cg1_err_abs / np.linalg.norm(b)

    cg2_err_abs = np.linalg.norm(A * x2 - b)
    cg2_err_rel = cg2_err_abs / np.linalg.norm(b)

    v1.append((itn1, t1, rng, info1))
    v2.append((itn2, t2, rng, info2))

    '''
    print()
    print("---------------------- PERFORMANCE --------------------")
    print("info1 = " + info1.__str__())
    print("info2 = " + info2.__str__())
    print()

    print("cg1_err_abs = " +  cg1_err_abs.__str__())
    print("cg1_err_rel = " + cg1_err_rel.__str__())
    print("itn1 = " + itn1.__str__())
    print("t1 = " + t1.__str__())
    print()

    print("cg2_err_abs = " +  cg2_err_abs.__str__())
    print("cg2_err_rel = " + cg2_err_rel.__str__())
    print("itn2 = " + itn2.__str__())
    print("t2 = " + t2.__str__())
    print()

    print("----------------- COMPONENTI NEL KERNEL ----------------")
    print()

    ones = np.ones(n)
    kcomp1 = np.dot(x1, ones)
    kcomp2 = np.dot(x2, ones)

    print("kcomp1 = " + kcomp1.__str__())
    print("kcomp2 = " + kcomp2.__str__())
    print()

    print("--------- DEVIAZIONI DAL SOLVER DI LIBRERIA -----------")
    print()
    print("x_dev_abs = " +  x_dev_abs.__str__())
    print("x_dev_rel = " + x_dev_rel.__str__())

    print("f_dev_abs = " +  f_dev_abs.__str__())
    print("f_dev_rel = " + f_dev_rel.__str__())
    '''

################################################################

def run(v1, v2):
    prefix = "mcf_generator/3000/netgen-3000-3-"
    suffix = "-b-b-s.dmx"

    for i in range (1, 6):
        file_path = prefix + i.__str__() + suffix
        for j in range (1, 11):
            for h in range (5):
                run_experiment(file_path, v1, v2, j*10)


v1 = []
v2 = []
run(v1, v2)

rel_time = []
avg = 0
for i in range (len(v1)):
     rel_time.append(v1[i][1]  / v2[i][1] - 1)
     avg += rel_time[-1]

avg /= len(v1)

print(v1)
print(avg)

    

    
