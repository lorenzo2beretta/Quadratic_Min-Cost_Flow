#  Si dichiara che il contenuto di questo file è in ogni sua parte
#  opera originale dell'autore.
#
#  Lorenzo Beretta, 536242, loribere@gmail.com

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import time


def conjugate_gradient(A, b, tol=1e-5, maxiter=None, callback=None):
    ''' This function implements the conjugate gradient algorithm 
    solving A * x = b for a symmetric matrix A.
    '''
    x = np.zeros(len(b))                 # current approximate solution
    r = np.array(b)                      # current residual value
    p = np.array(b)                      # current update direction
    rr = np.dot(r, r)                    # error squared norm
    itn = 0
    info = 0
    
    while np.sqrt(rr) > tol:

        if maxiter and itn > maxiter:    # method not converged
            info = maxiter
            break
        
        Ap = A * p                       # fast matrix-vector product
        alpha = rr / np.dot(p, Ap)       # step length
        x += alpha * p                   # current iterate

        rr_old = rr
        r -= alpha * Ap                  # current error
        rr = np.dot(r, r)

        beta = rr / rr_old               # step improvement
        p = r + beta * p                 # next update direction

        itn += 1
        callback(x)

    
    return x, info


def make_operator(edges, n):
    ''' This function returns a LinearOperator object performing
    the product x -> E^t * D^-1 * E * x
    '''
    edges = [(e[0], e[1], 1 / float(e[2])) for e in edges]
    
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

    A = LinearOperator((n, n), matvec=lin_op)
    return A 


def get_primal(edges, dual):
    ''' This function reconstruct primal solution given the dual one.
    '''
    primal = [(dual[e[1]] - dual[e[0]]) / e[2] for e in edges]
    return np.array(primal)


def solve(edges, b, tol=1e-5, algo=conjugate_gradient):
    ''' This function solves uncapacited and undirected quadratic 
    separable MCF, equivalent to the solution of the linear system 
    
    A * x = b,    with    A = E^t * D^-1 * E

    where E is the edge-node matrix and D is the diagonal matrix
    containing the edges' weights.

    It is parametric in the algorithm used to solve that linear
    system in order to compare different solutions.
    '''    
    A = make_operator(edges, len(b))  # defining linear operator

    itn = 0
    def callback(xk):  # counting iterations
        nonlocal itn
        itn += 1

    t0 = time.time()
    x, info = algo(A, b, tol=tol, callback=callback)  # solve A * x = b
    t1 = time.time()
    tspan = t1 - t0  # measuring elapsed time

    f = get_primal(edges, x)

    return x, f, itn, tspan, info


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
