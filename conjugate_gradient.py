import numpy as np

# TODO: correggi questa descrizione
''' ------------ Conjugate Gradient Algorithm -------------

This function implements the conjugate gradient algorithm [Reference:
Trefethen, Bau, Lecture 38] solving S * x = b for a strucutred symmetric matrix S.

We demanded the implementation of S * y products to a function multiply
since we do want to exploit the strucure of S to speed up the iteration time 
complexity.

In the end we applied this techinque to solve quadratic separable Min-Cost Flow
problem, reconducted to a linear system through a Poorman KKT approach.
In this case exploiting the structure of the S matrix is paramount since it
turns out to yield a significant saving in time complexity.

'''

def multiply(edg, d, y):
    ''' This function returns the product 
        
    E^t * diag(d) * E * y
    
    where E is the edge-node matrix of a graph whose edges are listed in edg.
    That is E[(i,j)][k] == (j == k) ? 1 : ( (i == k) ? -1 : 0 )
    '''
    n = len(d)
    for i in range (n):
        d[i] *= y[edg[i][1]] - y[edg[i][0]]

    for i in range (n):
        y[edg[i][1]] += d[i]
        y[edg[i][0]] -= d[i]

    return y
    
    
def conjugate_gradient(edg, d, b, threshold=1e-5):
    ''' Given S symmetric matrix this method solves Sx = b iteratively 
    using conjugate gradient.
    '''
    
    x = np.zeros(len(b))   # current approximate solution
    r = np.array[b]        # current residual value
    p = np.array[b]        # current update direction

    while True:
        prod = multiply(edg, d, p)
        alpha = np.dot(r, r)
        alpha /= np.dot(p, prod)
        x += alpha * p
        beta = np.dot(r, r)
        if np.sqrt(beta) < threshold:
            break
        r -= alpha * prod
        beta /= np.dot(r, r)
        p = r + beta * p

    return x

