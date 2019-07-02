from cg import solve
from cg import read_DIMACS
import scipy as sp
import numpy as np

file_path_1000 = 'mcf_generator/1000/netgen-1000-3-5-b-b-s.dmx'
file_path_2000 = 'mcf_generator/2000/netgen-2000-3-5-b-b-s.dmx'
file_path_3000 = 'mcf_generator/3000/netgen-3000-3-5-b-b-s.dmx'

edges, b = read_DIMACS(file_path_3000)

''' Inserting gaussian random weights '''

l = 0
r = 100000

edges = [(e[0], e[1], np.random.uniform(l, r)) for e in edges]

x1, f1, itn1, t1, info1 = solve(edges, b, 1e-10)

x2, f2, itn2, t2, info2 = solve(edges, b, 1e-10, sp.sparse.linalg.cg)

x_err_abs = np.linalg.norm(x1 - x2)
x_err_rel = x_err_abs / np.linalg.norm(x2)

f_err_abs = np.linalg.norm(f1 - f2)
f_err_rel = f_err_abs / np.linalg.norm(f2)

print("itn1 = " + itn1.__str__())
print("t1 = " + t1.__str__())

print("itn2 = " + itn2.__str__())
print("t2 = " + t2.__str__())

print("x_err_abs = " +  x_err_abs.__str__())
print("x_err_rel = " + x_err_rel.__str__())

print("f_err_abs = " +  f_err_abs.__str__())
print("f_err_rel = " + f_err_rel.__str__())
