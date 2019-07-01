from conjugate_gradient import conjugate_gradient
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
            b = [0] * int(token[2])
            
        if line[:1] == 'n':
            token = line.split()
            b[int(token[1]) - 1] = float(token[2])

        if line[:1] == 'a':
            token = line.split()
            u = int(token[1]) - 1
            v = int(token[2]) - 1
            c = float(token[5])
            edges.append((u, v, c))

    return edges, b


''' This is a preliminary test, just to check that the method converges
to something meningful.
'''

file_path_1000 = 'mcf_generator/1000/netgen-1000-3-5-b-b-s.dmx'
file_path_2000 = 'mcf_generator/2000/netgen-2000-3-5-b-b-s.dmx'
file_path_3000 = 'mcf_generator/3000/netgen-3000-3-5-b-b-s.dmx'

edges, b = read_DIMACS(file_path_2000)

''' Inserting gaussian random weights '''

mu = 0 
sigma = 0.00001

# edges = [ (e[0], e[1], np.random.normal(mu, sigma)) for e in edges]

l = -100
r = 100

edges = [ (e[0], e[1], np.random.uniform(l, r)) for e in edges]
   
f = conjugate_gradient(edges, b)

# print(f)



            
