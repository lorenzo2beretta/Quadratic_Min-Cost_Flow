from cg import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
  

''' Generating problem's data. ''' 
file_path = 'graph'
edges, n = read_DIMACS(file_path)
rad = 100
D = [np.exp(np.random.uniform(-rad, rad)) for e in edges]

rad_b = 10
b = [np.random.uniform(-rad_b, rad_b) for i in range (n)]
proj = np.ones(n)
proj *= np.dot(proj, b) / n
b -= proj

A = make_operator(edges, D, n)  # defining linear operator

Apr, bpr = precondition(edges, D, b)


''' Setting custom parameters '''
maxiter = 1000
tol = 1e-5

def run(A, b, algo):
    res = []
    def callback(xk):
        nonlocal res
        r = np.linalg.norm(A * xk - b)
        res.append(r)
        
    t0 = time.time()
    x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback)
    t1 = time.time()
    tspan = t1 - t0  # measuring elapsed time
    itn = len(res)
    
    return x, info, itn, tspan, res

print( n, len(edges), rad, rad_b)

# comparison

x, info, itn, tspan, res = run(A, b, my_cg)
print('A:', itn , tspan)

print(np.linalg.norm(A * x - b)) 

# plots
res = np.log(np.array(res))
plt.plot(res)

x, info, itn, tspan, res = run(Apr, bpr, my_cg)
print('Apr:', itn , tspan)

print(np.linalg.norm(A * x - b))

# plots
res = np.log(np.array(res))
plt.plot(res)

plt.show()







# eigenvalues plot
'''
M = A * np.identity(n)

diag = [M[i][i] for i in range(n)]

diag.sort()
diag = np.log(diag)
plt.plot(diag)
plt.show()
'''

'''
eig = np.linalg.eigvals(M)


eig = [np.linalg.norm(e) for e in eig]
eig.sort()
print(eig[0:40])

eig = np.log(eig)
plt.plot(eig)
plt.show()
'''

    
