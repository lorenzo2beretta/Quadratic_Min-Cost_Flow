from cg import solve
import scipy
import numpy as np

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


''' This is a preliminary test, just to check that the method converges
to something meningful.
'''

file_path_1000 = 'mcf_generator/1000/netgen-1000-3-5-b-b-s.dmx'
file_path_2000 = 'mcf_generator/2000/netgen-2000-3-5-b-b-s.dmx'
file_path_3000 = 'mcf_generator/3000/netgen-3000-3-5-b-b-s.dmx'

edges, b = read_DIMACS(file_path_1000)

''' Inserting gaussian random weights '''

l = 0
r = 100

edges = [(e[0], e[1], np.random.uniform(l, r)) for e in edges]
   
f1 = solve(edges, b)
f2 = solve(edges, b, algo=scipy.sparse.linalg.cg)

err_abs = np.linalg.norm(f1 - f2)
err_rel = err_abs / np.linalg.norm(f2)

print("err_abs = " +  err_abs.__str__())
print('err_rel = ' + err_rel.__str__())


            
