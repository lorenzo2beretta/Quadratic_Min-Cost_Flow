from cg import *
from scipy.sparse.linalg import cg
import numpy as np

file_path_1000 = 'mcf_generator/1000/netgen-1000-3-5-b-b-s.dmx'
file_path_2000 = 'mcf_generator/2000/netgen-2000-3-5-b-b-s.dmx'
file_path_3000 = 'mcf_generator/3000/netgen-3000-3-5-b-b-s.dmx'

edges, b = read_DIMACS(file_path_3000)

edges = [(e[0], e[1], np.exp(np.random.normal(0, 5))) for e in edges]

# edges[0] = (edges[0][0], edges[0][1], 0.000000000001)

x1, f1, itn1, t1, info1 = solve(edges, b, maxiter=500)

x2, f2, itn2, t2, info2 = solve(edges, b, maxiter=500, algo=cg)

x_err_abs = np.linalg.norm(x1 - x2)
x_err_rel = x_err_abs / np.linalg.norm(x2)

f_err_abs = np.linalg.norm(f1 - f2)
f_err_rel = f_err_abs / np.linalg.norm(f2)

n = len(b)
A = make_operator(edges, n) 

cg1_err_abs = np.linalg.norm(A * x1 - b)
cg1_err_rel = cg1_err_abs / np.linalg.norm(b)

cg2_err_abs = np.linalg.norm(A * x2 - b)
cg2_err_rel = cg2_err_abs / np.linalg.norm(b)

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



''' PSEUDO ERRORI (i.e. deviazioni dal solver di libreria)
print("x_err_abs = " +  x_err_abs.__str__())
print("x_err_rel = " + x_err_rel.__str__())

print("f_err_abs = " +  f_err_abs.__str__())
print("f_err_rel = " + f_err_rel.__str__())
'''

