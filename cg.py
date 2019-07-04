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


def make_operator(edges, n):
    ''' This function returns a LinearOperator object performing
    the product x -> E * D^-1 * E^t * x where E is the edge-node
    matrix and D is the diagonal matrix encoding edges' weights.

    Parameters:

    edges: {list}
        It is a list of triples (u, v, w) where u is the start
        node, v the end node and w the quadratic weight associated.

    n: {integer}
        It is the number of nodes, or equivalenty the dimention of
        the input and output spaces of the operator created.

    Returns:

    A: {LinearOperator}
        A is a LinearOperator performing x -> (E * D^-1 * E^t) * x.
    '''
    edges = [(e[0], e[1], 1 / float(e[2])) for e in edges]
    
    def matvec(x):
        ''' Fast matrix-vector multiplication exploiting the structure
        of E, achieving O(|A|) time complexity.
        '''
        res = np.zeros(len(x))
        for e in edges:
            diff = e[2] * (x[e[0]] - x[e[1]])
            res[e[0]] += diff
            res[e[1]] -= diff

        return res

    A = LinearOperator((n, n), matvec=matvec)
    return A 


def get_primal(edges, x):
    ''' This function reconstruct primal solution given the dual one.
    Basically it solves f^t * D = x * E that is the KKT-G condition.

    Parameters:
    
    edges: {list}
        It is a list of triples (u, v, w) where u is the start
        node, v the end node and w the quadratic weight associated.

    x: {array, matrix}
        The dual solution.
    '''
    f = [(x[e[1]] - x[e[0]]) / e[2] for e in edges]
    return np.array(f)


def solve(edges, b, maxiter=None, tol=1e-5, algo=my_cg):
    ''' This function solves uncapacited and undirected quadratic 
    separable MCF, equivalent to the solution of the linear system 
    
    A * x = b,    with    A = E * D^-1 * E^t

    where E is the edge-node matrix and D is the diagonal matrix
    containing the edges' weights.

    It is parametric in the algorithm used to solve that linear
    system in order to compare different solutions.

    Parameters:

    edges: {list}
        It is a list of triples (u, v, w) where u is the start
        node, v the end node and w the quadratic weight associated.

    b: {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
        It encodes the flow balance conditionds defining sinks and 
        sources. 

    Returns:

    x: {array, matrix}
        The converged dual solution, provided by linear solver.
    
    f: {array, matrix}
        The primal solution (i.e. the optimal MCF).

    itn: {integer}
        Number of iteration required by the solver.
    
    tspan: {float}
        Time required to solve the linear system.

    info: {integer}
        Provides linear solver convergence information:
        0 : successful exit 
        >0 : convergence not achieved, returns number of iterations

    Other Parameters:

    maxiter: {integer}
        Maximum number of iterations for linear solver. Iteration 
        will stop after maxiter steps even if the specified tolerance 
        has not been achieved.

    tol: {float}
        Solver convergence condition is norm(residual) <= tol*norm(b).  
    '''    
    A = make_operator(edges, len(b))  # defining linear operator

    itn = 0
    def callback(xk):  # counting iterations
        nonlocal itn
        itn += 1

    t0 = time.time()
    x, info = algo(A, b, maxiter=maxiter, tol=tol, callback=callback)
    t1 = time.time()
    tspan = t1 - t0  # measuring elapsed time

    f = get_primal(edges, x)

    return x, f, itn, tspan, info


def read_DIMACS(file_path):
    ''' This method reads topology, costs and balance vector of an 
    undirected graph form a file following the DIMACS Min-Cost flow 
    conventions.
    
    Parameters:

    file_path: {string}
        The path form current directory to the DIMACS compliant file
        encoding the MCF graph.

    Returns:

    edges: {list}
        It is a list of triples (u, v, w) where u is the start
        node, v the end node and w the quadratic weight associated.

    b: {array, matrix}
        It encodes the flow balance conditionds defining sinks and 
        sources.
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
