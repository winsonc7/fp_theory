import numpy as np
from scipy.sparse import csr_matrix

A = 5       # forward rate
B = 2       # backward rate
N = 1000    # matrix size
T = 1       # time
K = 1000    # num terms to sum

def generate_rate_matrix(a, b, n):
    rate_matrix = csr_matrix((n,n), dtype=float)
    for i in range(n):
        if i + 1 < n - 1:   # bounds checking
            rate_matrix[i][i+1] = a
        if i != 0:          # bounds checking
            rate_matrix[i][i-1] = b
        rate_matrix[i][i] = sum(rate_matrix[i]) * -1
    return rate_matrix

def 







r_matrix = generate_rate_matrix(A, B, N)


