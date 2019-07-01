import numpy as np
import scipy as sp 

def conjugate_gradient(A, b, tol=1e-5):
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

    return x


def solve(edges, b, threshold=1e-5, algo=conjugate_gradient):
    
    edges = [(e[0], e[1], 1 / float(e[2])) for e in edges]
    b = [float(z) for z in b]
    
    def lin_op(x):
        res = np.zeros(len(x))
        for e in edges:
            diff = e[2] * (x[e[0]] - x[e[1]])
            res[e[0]] += diff
            res[e[1]] -= diff

        return res

    n = len(b)
    A = sp.sparse.linalg.LinearOperator((n, n), matvec=lin_op)
    x = algo(A, b, tol=threshold)              # solve A * x = b

    if algo == sp.sparse.linalg.cg:  # off-the-shelf CG nasty signature
        x = x[0]
        
    f = [e[2] * (x[e[1]] - x[e[0]]) for e in edges]  # primal solution
    return np.array(f)



