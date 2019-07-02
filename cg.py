#  Si dichiara che il contenuto di questo file è in ogni sua parte
#  opera originale dell'autore.
#
#  Lorenzo Beretta, 536242, loribere@gmail.com

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import time


def conjugate_gradient(A, b, tol=1e-5, callback=None):
    ''' This function implements the conjugate gradient algorithm 
    solving A * x = b for a symmetric matrix A.
    '''
    x = np.zeros(len(b))                 # current approximate solution
    r = np.array(b)                      # current residual value
    p = np.array(b)                      # current update direction
    rr = np.dot(r, r)                    # error squared norm
    
    while np.sqrt(rr) > tol:
        
        Ap = A * p                       # fast matrix-vector product
        alpha = rr / np.dot(p, Ap)       # step length
        x += alpha * p                   # current iterate

        rr_old = rr
        r -= alpha * Ap                  # current error
        rr = np.dot(r, r)

        beta = rr / rr_old               # step improvement
        p = r + beta * p                 # next update direction

        callback(x)

    return x


def solve(edges, b, tol=1e-5, algo=conjugate_gradient):
    ''' This function solves uncapacited and undirected quadratic 
    separable MCF, equivalent to the solution of the linear system 
    
    E * D^-1 * E^t * x = b

    where E is the edge-node matrix and D is the diagonal matrix
    containing the edges' weights.

    It is parametric in the algorithm used to solve that linear
    system in order to compare different solutions.
    '''    
    edges = [(e[0], e[1], 1 / float(e[2])) for e in edges]
    b = [float(z) for z in b]
    
    def lin_op(x):
        ''' Fast matrix-vector multiplication exploiting the structure
        of E, achieving O(|A|) time complexity.
        '''
        res = np.zeros(len(x))
        for e in edges:
            diff = e[2] * (x[e[0]] - x[e[1]])
            res[e[0]] += diff
            res[e[1]] -= diff

        return res

    itn = 0
    def callback(xk):  # callback function to count iterations
        nonlocal itn
        itn += 1

    n = len(b)
    A = LinearOperator((n, n), matvec=lin_op)   # define x -> A * x

    t0 = time.time()
    x = algo(A, b, tol=tol, callback=callback)  # solve A * x = b
    t1 = time.time()
    tspan = t1 - t0  # measuring elapsed time
    
    if algo == sp.sparse.linalg.cg:  # off-the-shelf CG nasty signature
        x = x[0]
        
    f = [e[2] * (x[e[1]] - x[e[0]]) for e in edges]  # primal solution

    return x, np.array(f), itn, tspan
