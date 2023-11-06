import numpy as np
import math
from scipy.sparse import csr_matrix, eye, diags

A = 5       # forward rate
B = 11       # backward rate: more stable if B > A
N = 10      # matrix size: assumes N > 1
T = 2       # time
K = 200     # num terms to sum, on select examples K=200 seems pretty stable

# Output files are labeled like the following: standard/unif_A_B_N_T_K.csv

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

standard_answer = standard_exponential(q_matrix, T, K)
np.savetxt("outputs/s_5_11_10_2_200.csv", standard_answer.toarray(), delimiter=',', fmt='%f')

unif_answer = unif_exponential(q_matrix, A, B, T, K)
np.savetxt("outputs/u_5_11_10_2_200.csv", unif_answer.toarray(), delimiter=',', fmt='%f')