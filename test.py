from conjugate_gradient import conjugate_gradient

edges = []
edges.append((0, 1, 0.1))
edges.append((1, 2, 0.2))
edges.append((2, 3, 0.3))
edges.append((0, 3, 10.0))

b = [-10, 0, 0, 10]
b = [float(z) for z in b]

f = conjugate_gradient(edges, b, 1e-10)

print(f)
    
