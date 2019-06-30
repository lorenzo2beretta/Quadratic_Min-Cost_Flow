import numpy as np

''' ------------ Conjugate Gradient Algorithm -------------

This function implements the conjugate gradient algorithm [Reference:
Trefethen, Bau, Lecture 38] solving S * x = b for a strucutred symmetric 
matrix S.

In particular we applied this techinque to solve quadratic separable 
Min-Cost Flow problem, reconducted to a linear system through a dual 
approach exploiting KKT conditions.

This led to a formulation of the original problem as a linar system of
the form (E^t * D^-1 * E) * x = b where E is the edge-node matrix of
the original undirected and uncapacited flow graph and D is a diagonal
positive definite matrix containing quadratic coefficients.

We took advantage of the particular strucutre of the symmetric matrix
and implemented the function multiply performing matrix-vector product
in O(m) where m = |A| in the graph G = (A, E). Thus we achieved a 
time coplexity of O(m) per iteration of conjugate gradient. 
'''

def multiply(edges, x):
# TODO: write better documentation
    ''' This function returns the product 
        
    E^t * diag(edg[2]) * E * x

    where E is the edge-node matrix of a graph whose edges are listed 
    in edge, that is 
    
    E[(i,j)][k] == (j == k) ? 1 : ( (i == k) ? -1 : 0 )
    
    '''
    res = np.zeros(len(x))

    for e in edges:
        diff = e[2] * (x[e[0]] - x[e[1]])
        res[e[0]] += diff
        res[e[1]] -= diff
                
    return res
    
    
def conjugate_gradient(edges, b, threshold=1e-5):
    ''' Iteratively solves the linear system

    E^t * diag(d)^-1 * E * y
    
    '''
    # we employ weights' inverses instead of weights
    edges = [(e[0], e[1], 1 / e[2]) for e in edges]
    
    x = np.zeros(len(b))   # current approximate solution
    r = np.array(b)        # current residual value
    p = np.array(b)        # current update direction
    beta = np.dot(r, r)

    while beta != 0:
        prod = multiply(edges, p)
        alpha = np.dot(r, r)
        alpha /= np.dot(p, prod)
        x += alpha * p
        beta = np.dot(r, r)
        if np.sqrt(beta) < threshold:
            break
        r -= alpha * prod
        beta = np.dot(r, r) / beta
        p = r + beta * p

    f = [ e[2] * (x[e[1]] - x[e[0]]) for e in edges]
    return f

