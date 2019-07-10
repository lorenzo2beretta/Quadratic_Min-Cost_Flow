from cg import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import gmres

''' Generating problem's data. ''' 

file_path = 'graph'
edges, n = read_DIMACS(file_path)

rad_D = 50
D = [np.exp(np.random.uniform(-rad_D, rad_D)) for e in edges]

# trick to sample b s.t. np.ones(len(b))^t * b == 0
rad_b = 10
b = [np.random.uniform(-rad_b, rad_b) for i in range (n)]
proj = np.ones(n)
proj *= np.dot(proj, b) / n
b -= proj

A = make_operator(edges, D, n)  # defining linear operator
M = make_jacobi_prec(edges, D, n)  # defining Jacobi preconditioner


''' Setting custom parameters '''

def run(A, b, algo, tol=1e-5, maxiter=10000, M=None):
    res = []
    if algo == gmres:
        def callback(rk):
            r = np.linalg.norm(rk)
            res.append(r)
    else:
        def callback(xk):
            r = np.linalg.norm(A * xk - b)
            res.append(r)

    t0 = time.time()
    if M:
        x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback, M=M)
    else:
        x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback)
    t1 = time.time()

    tspan = t1 - t0  # measuring elapsed time
    itn = len(res)   # counting iterations
    acc = np.linalg.norm(A * x - b) / np.linalg.norm(b)
    
    return x, acc, itn, tspan, res

print( n, len(edges), rad_D, rad_b)


# ---------------- COMPARISON -------------------

# standard cg
x, acc, itn, tspan, res = run(A, b, my_cg)
print('standard cg:', itn , tspan, acc)
res = np.log(np.array(res))
plt.title(r'$n$ = '+str(n)+' $, m$ = '+str(len(edges))+' $, rad_D$ = '+str(rad_D))
plt.xlabel(r'$k$-th iteration')
plt.ylabel(r'$\log\left(\left||r_k\right||_2\right)$')
plt.plot(res)

'''
# preconditioned cg
x, acc, itn, tspan, res = run(A, b, cg, M=M)
print('preconditioned cg:', itn , tspan, acc)
res = np.log(np.array(res))
plt.plot(res)
'''

# standard GMRES
x, acc, itn, tspan, res = run(A, b, gmres)
print('standard GMRES:', itn , tspan, acc)
res = np.log(np.array(res))
plt.plot(res)
'''
# preconditioned GMRES
x, acc, itn, tspan, res = run(A, b, gmres, M=M)
print('preconditioned GMRES:', itn , tspan, acc)
res = np.log(np.array(res))
plt.plot(res)
'''

# plt.show()  # show plots



'''
# eigenvalues plot

M = A * np.identity(n)

# Questo serve per giustificare l'uso di Jacobi

diag = [M[i][i] for i in range(n)]

diag.sort()
diag = np.log(diag)
plt.plot(diag)
plt.xlabel(r'$i$-th diagonal entry')
plt.ylabel(r'$\log\left(\left|A_{i, i}\right|\right)$')
plt.title('Norm of $A$\' diagonal entries')
plt.show()


eig = np.linalg.eigvals(M)


eig = [np.linalg.norm(e) for e in eig]
eig.sort()
# print(eig[0:40])

eig = np.log(eig)
plt.plot(eig)
plt.xlabel(r'$i$-th eigenvalue')
plt.ylabel(r'$\log\left(\left|\lambda_i\right|\right)$')
plt.title('Density = 10%, $rad_D$ = 100')
plt.show()


'''    
