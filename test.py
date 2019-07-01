from cg import solve
import scipy as sp
import numpy as np
import time

def read_DIMACS(file_path):
    ''' This method reads the topology and costs of an undirected 
    graph form a file following the DIMACS Min-Cost flow convention.
    '''
    file = open(file_path, 'r')
    edges = []
    
    while True:
        line = file.readline()
        if line == '':
            break

        if line[:1] == 'c':
            continue

        if line[:1] == 'p':
            token = line.split()
            n = int(token[2]) + 1
            b = [0] * n
            
        if line[:1] == 'n':
            token = line.split()
            b[int(token[1])] = float(token[2])

        if line[:1] == 'a':
            token = line.split()
            u = int(token[1])
            v = int(token[2])
            c = float(token[5])
            edges.append((u, v, c))

    return edges, b


def time_wrap(function):
    ''' Funtion wrapper to measure time elapsed '''
    def ret(*args):
        t0 = time.time()
        y = function(*args)
        t1 = time.time()
        return y, t1 - t0
    return ret


file_path_1000 = 'mcf_generator/1000/netgen-1000-3-5-b-b-s.dmx'
file_path_2000 = 'mcf_generator/2000/netgen-2000-3-5-b-b-s.dmx'
file_path_3000 = 'mcf_generator/3000/netgen-3000-3-5-b-b-s.dmx'

edges, b = read_DIMACS(file_path_3000)

''' Inserting gaussian random weights '''

l = 100000000
r = 1000000000

edges = [(e[0], e[1], np.random.uniform(l, r)) for e in edges]

solve = time_wrap(solve)

ret1, t1 = solve(edges, b, 1e-10)
f1, itn1 = ret1
ret2, t2 = solve(edges, b, 1e-10, sp.sparse.linalg.cg)
f2, itn2 = ret2

print("t1 = " + t1.__str__())
print("t2 = " + t2.__str__())

print("itn1 = " + itn1.__str__())
print("itn2 = " + itn2.__str__())

err_abs = np.linalg.norm(f1 - f2)
err_rel = err_abs / np.linalg.norm(f2)

print("err_abs = " +  err_abs.__str__())
print("err_rel = " + err_rel.__str__())


            
