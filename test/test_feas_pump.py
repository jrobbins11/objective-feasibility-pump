import objective_feasibility_pump as ofp
import numpy as np
from scipy import sparse

def random_sparse_matrix(m, n, density, val_range):
    assert(density > 0 and density <= 1)
    assert(m > 0 and n > 0)
    
    max_nnz = m*n
    nnz = int(np.round(max_nnz * density))

    # triplets
    rows = []
    cols = []
    vals = []
    for _ in range(nnz):
        rows.append(np.round(np.random.random()*(m-1)))
        cols.append(np.round(np.random.random()*(n-1)))
        vals.append(np.random.random()*(2*val_range) - val_range)

    # matrix
    return sparse.csc_matrix((vals, (rows, cols)), shape=(m,n))


def random_vector(m, val_range):
    return np.random.rand(m)*val_range - val_range
    

### main script ###

# dimensions
m = 1000
n = 2300
nb = 600
density = 0.003

# problem
c = random_vector(n, 10.)
x_l = np.zeros(n)
x_u = np.ones(n)
A = random_sparse_matrix(m, n, density, 1.0)
A_l = -2.*np.ones(m)
A_u = 2.*np.ones(m)
bins = [i for i in range(n-nb, n)]

# solve
settings = ofp.OFP_Settings()
settings.max_iter = 10000
settings.alpha0 = 1.
settings.t_max = 10.

OFP = ofp.OFP_Solver(c, A, A_l, A_u, x_l, x_u, bins, settings=settings)
success = OFP.solve()
sol = OFP.get_solution()
print(f'successful? {success}')
print(sol)

info = OFP.get_info()
print(f'iter: {info.iter}')
print(f'restarts: {info.restarts}')
print(f'perturbations: {info.perturbations}')
print(f'runtime: {info.runtime}')
print(f'feasible: {info.feasible}')
print(f'alpha: {info.alpha}')