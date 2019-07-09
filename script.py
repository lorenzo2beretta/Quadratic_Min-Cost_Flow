from cg import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
  

''' Generating problem's data. ''' 
file_path = 'graph'
edges, n = read_DIMACS(file_path)

rad_D = 500
D = [np.exp(np.random.uniform(-rad_D, rad_D)) for e in edges]

# trick to sample b s.t. np.ones(len(b))^t * b == 0
rad_b = 10
b = [np.random.uniform(-rad_b, rad_b) for i in range (n)]
proj = np.ones(n)
proj *= np.dot(proj, b) / n
b -= proj

A = make_operator(edges, D, n)  # defining linear operator

# Apr, bpr = precondition(edges, D, b)


''' Setting custom parameters '''
maxiter = 1000
tol = 1e-5

def run(A, b, algo, isPrec=False):
    res = []
    def callback(xk):
        nonlocal res
        r = np.linalg.norm(A * xk - b)
        res.append(r)
        
    t0 = time.time()
    if isPrec:
        M = make_preconditioner(edges, D, n)
        x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback, M=M)
    else:
        x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback)
    t1 = time.time()
    tspan = t1 - t0  # measuring elapsed time
    itn = len(res)
    
    return x, info, itn, tspan, res

print( n, len(edges), rad_D, rad_b)

# comparison

# standard
'''
x, info, itn, tspan, res = run(A, b, my_cg)
print('standard:', itn , tspan)
res = np.log(np.array(res))
plt.plot(res)
'''
# preconditioned
x, info, itn, tspan, res = run(A, b, cg, isPrec=True)
print('preconditioned:', itn , tspan)
res = np.log(np.array(res))
plt.plot(res)

print(np.linalg.norm(A * x - b))

# show plots
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

    
