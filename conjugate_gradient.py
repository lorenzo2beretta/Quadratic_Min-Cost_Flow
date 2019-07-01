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
    ''' This function returns the product 
        
    E^t * D * E * x

    where E is the edge-node matrix of a graph whose edges are listed 
    in edges, that is 
    
    E[h][k] == (edges[h][0] == k) ? -1 : ( (edges[h][1] == k) ? 1 : 0 )
    
    and D = diag(D) is such that D_{i,i} == edges[i][2]
    '''
    res = np.zeros(len(x))

    for e in edges:
        diff = (x[e[0]] - x[e[1]]) / e[2]
        res[e[0]] += diff
        res[e[1]] -= diff
                
    return res
    
    
def conjugate_gradient(edges, b, threshold=1e-10):
    ''' Uses Conjugate Gradient method to solve the linear system

    E^t * D^-1 * E * x = b

    where E is the edge-node matrix of a graph whose edges are listed 
    in edges, that is 
    
    E[h][k] == (edges[h][0] == k) ? -1 : ( (edges[h][1] == k) ? 1 : 0 )
    
    and D = diag(D) is such that D_{i,i} == edges[i][2]
    '''
        
    x = np.zeros(len(b))                 # current approximate solution
    r = np.array([float(z) for z in b])  # current residual value
    p = np.array([float(z) for z in b])  # current update direction
    beta = np.dot(r, r)                  # error squared norm

    while beta != 0:
        prod = multiply(edges, p)        # fast matrix-vector product
        
        alpha = np.dot(r, r)
        alpha /= np.dot(p, prod)         # step length

        x += alpha * p                   # current iterate
        
        beta = np.dot(r, r)

        # TODO: comment ###################################
        # print(np.sqrt(beta))  # prints eudclidean norm
        # print(np.sqrt(np.dot(r, multiply(edges, r))))  # prints S-norm 
        tmp = np.dot(r, multiply(edges, r))
        ###################################################
        
        if np.sqrt(beta) < threshold:    # stopping criterion
            break

        r -= alpha * prod                # current error
        
        beta = np.dot(r, r) / beta
        
        # TODO: comment ##########################
        # print(beta)
        tmp1 = np.dot(r, multiply(edges, r))
        print(np.sqrt(tmp1 / tmp))
        #########################################
        
        p = r + beta * p                 # next update direction

    f = [ e[2] * (x[e[1]] - x[e[0]]) for e in edges]  # primal solution
    return f

