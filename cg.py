#  Si dichiara che il contenuto di questo file e in ogni sua parte
#  opera originale dell'autore.
#
#  Lorenzo Beretta, 536242, loribere@gmail.com

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import time


def my_cg(A, b, tol=1e-5, maxiter=None, callback=None):
    ''' This function implements the conjugate gradient algorithm 
    solving A * x = b for a symmetric semipositive-definite matrix A.

    Parameters:

    A: {sparse marix, dense matrix, LinearOperator}
        It is the symmetric linear operator to invert. It must be able
        to perform A * x for an array or matrix x of appropriate size.
    
    b: {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    
    Returns:
    
    x: {array, matrix}
        The converged solution.
    
    info: {integer}
        Provides convergence information:
        0 : successful exit 
        >0 : convergence not achieved, returns number of iterations

    Other Parameters:

    tol: {float}
        Tolerances for convergence, norm(residual) <= tol*norm(b)

    maxiter: {integer}
        Maximum number of iterations. Iteration will stop after maxiter 
        steps even if the specified tolerance has not been achieved.

    callback: {function}
        User-supplied function to call after each iteration. It is called 
        as callback(xk), where xk is the current solution vector.
    '''
    x = np.zeros(len(b))                 # current approximate solution
    r = np.array(b)                      # current residual value
    p = np.array(b)                      # current update direction
    rr = np.dot(r, r)                    # error squared norm
    tol *= np.sqrt(rr)
    itn = 0
    
    while np.sqrt(rr) > tol:

        if maxiter and itn >= maxiter:   # method not converged
            return x, maxiter
        
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

    
    return x, 0


def make_operator(edges, D, n):
    ''' This function returns a LinearOperator object performing
    the product x -> E * D^-1 * E^t * x where E is the edge-node
    matrix and D is a diagonal positive definite matrix.

    Parameters:

    edges: {list}
        It is a list of pairs (u, v) where u is the start node 
        and v the end node.

    D: {list, array}
        It is the list of diagonal elements of matrix D.
    
    n: {integer}
        It is the number of nodes, or equivalenty the dimention of
        the input and output spaces of the operator created.

    Returns:

    A: {LinearOperator}
        A is a LinearOperator performing x -> (E * D^-1 * E^t) * x.
    '''
    D = [1 / float(d) for d in D]  # performing divisions just once
    
    def matvec(x):
        ''' Fast matrix-vector multiplication exploiting the structure
        of E^t * D * E. It employs exactly m multiplication and 3m
        additions.
        '''
        res = np.zeros(len(x))
        for e, d in zip(edges, D):
            diff = d * (x[e[0]] - x[e[1]])
            res[e[0]] += diff
            res[e[1]] -= diff

        return res

    A = LinearOperator((n, n), matvec=matvec)
    return A
'''
def precondition(edges, D, b):
    n = len(b)
    diag = [0] * n
    for e, d in zip(edges, D):
        d = 1 / float(d)
        diag[e[0]] += d
        diag[e[1]] += d

    diag = [1 / d for d in diag]
    A = make_operator(edges, D, n)

    def matvec(x):
        res = A * x
        res = [r * d for r, d in zip(res, diag)]
        return res

    Apr = LinearOperator((n, n), matvec=matvec)
    b = [z * d for z, d in zip(b, diag)]
    return Apr, b
'''

def make_preconditioner(edges, D, n):
    prec = [0] * n
    for e, d in zip(edges, D):
        d = 1 / float(d)
        prec[e[0]] += d
        prec[e[1]] += d

    prec = [1 / p for p in prec]

    def matvec(x):
        res = [z * p for z, p in zip(x, prec)]
        return res

    M = LinearOperator((n, n), matvec=matvec)
    return M
        

def read_DIMACS(file_path):
    ''' This method reads the topology of a digraph form a file 
    following the DIMACS Min-Cost flow conventions.
    
    Parameters:

    file_path: {string}
        The path form current directory to the DIMACS compliant file
        encoding the MCF graph.

    Returns:

    edges: {list}
        It is a list of triples (u, v) where u is the start node and 
        v the end node.
    
    n: {integer}
        It is the number of nodes in graph.

    '''
    file = open(file_path, 'r')
    edges = []
    line = file.readline()
    
    while line != '':
        if line[:1] == 'p':
            token = line.split()
            n = int(token[2])
            
        if line[:1] == 'a':
            token = line.split()
            u = int(token[1]) - 1
            v = int(token[2]) - 1
            edges.append((u, v))

        line = file.readline()
            
    return edges, n
