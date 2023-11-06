import numpy as np
import math
from scipy.sparse import csr_matrix, eye, diags
from scipy.linalg import expm

A = 10       # forward rate
B = 13       # backward rate: more stable if B > A
N = 15      # matrix size: assumes N > 1
T = 10       # time
K = 200     # num terms to sum

# Output files are labeled like the following: standard/unif/benchmark_A_B_N_T_K.csv

def generate_MM1_rate_matrix(a, b, n):
    data = [a*-1, a]            # first row
    row_indices = [0, 0]
    column_indices = [0, 1]
    for i in range(1,n):
        vals = []
        if i < n - 1:
            vals = [b, -1*(a+b), a]
        else:
            vals = [b, b*-1]    # last row
        data = data + vals
        row_indices = row_indices + [i for j in range(len(vals))]
        column_indices = column_indices + [(i-1) + j for j in range(len(vals))]
    rate_matrix = csr_matrix((data, (row_indices, column_indices)), shape=(n, n))
    return rate_matrix

def standard_exponential(q, t, k):
    new_q = t * q
    curr_q = eye(new_q.shape[0], format='csr')          # identity matrix
    old = curr_q
    sum = curr_q
    for i in range(1, k):
        curr_q = (old @ new_q) / i
        old = curr_q
        sum += curr_q
    return sum

def unif_exponential(matrix, a, b, t, k):
    largest = a + b
    if matrix.shape[0] < 3:
        largest = max(a, b)
    add_m = diags([largest], [0], shape=matrix.shape, format='csr')
    shifted_m = matrix + add_m
    e_const = math.exp(-1*largest*t)
    unif_m = standard_exponential(shifted_m, t, k) * e_const
    return unif_m

q_matrix = generate_MM1_rate_matrix(A, B, N)

q_t = q_matrix * T
dense_m = q_t.toarray()
benchmark_answer = expm(dense_m)
# np.savetxt("outputs/b_10_13_10_15.csv", benchmark_answer, delimiter=',', fmt='%f')

standard_answer = standard_exponential(q_matrix, T, K)
# np.savetxt("outputs/s_10_13_15_20_200.csv", standard_answer.toarray(), delimiter=',', fmt='%f')

unif_answer = unif_exponential(q_matrix, A, B, T, K)
np.savetxt("outputs/u_10_13_15_10_200.csv", unif_answer.toarray(), delimiter=',', fmt='%f')