import numpy as np
import math
from scipy.sparse import csr_matrix, eye, save_npz

A = 4       # forward rate
B = 2       # backward rate
N = 3       # matrix size: assumes N > 1
T = 1       # time
K = 100      # num terms to sum: assumes K > 1


def generate_rate_matrix(a, b, n):
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

def generate_const_matrix(num, n):
    data = [num for i in range(n**2)]
    row_indices = []
    column_indices = []
    for i in range(n):
        row_indices += [i for j in range(n)]
        column_indices += [j for j in range(n)]
    m = csr_matrix((data, (row_indices, column_indices)), shape=(n, n))
    return m

def factorial_memo(n):
    memo = [1]
    for i in range(1, n):
        memo.append(memo[-1] * i)
    return memo

def standard_exponential(q, t, k):
    f_memo = factorial_memo(k)      # calculates factorials beforehand, idk felt like it'd be faster
    new_q = t * q
    curr_q = eye(new_q.shape[0], format='csr')          # identity matrix
    sum = curr_q
    for i in range(1, k):
        curr_q = curr_q @ new_q         # multiplication of sparse matrices
        sum += curr_q / f_memo[i]
    return sum

def unif_exponential(matrix, a, b, t, k):
    larger = max(a, b)              # csr matrices can't directly add non-zero scalars
    add_m = generate_const_matrix(larger, matrix.shape[0])
    shifted_m = matrix + add_m
    e_const = math.exp(-1*larger*t)
    unif_m = standard_exponential(shifted_m, t, k) * e_const
    return unif_m

q_matrix = generate_rate_matrix(A, B, N)

# standard_answer = standard_exponential(q_matrix, T, K)
# np.savetxt("p_matrix_standard.csv", standard_answer.toarray(), delimiter=',', fmt='%f')

unif_answer = unif_exponential(q_matrix, A, B, T, K)
np.savetxt("p_matrix_unif.csv", unif_answer.toarray(), delimiter=',', fmt='%f')