import numpy as np
from scipy.sparse import csr_matrix, eye, save_npz

A = 4       # forward rate
B = 2       # backward rate
N = 10       # matrix size: assumes N > 1
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

q_matrix = generate_rate_matrix(A, B, N)
standard_answer = standard_exponential(q_matrix, T, K)
save_npz("pmatrix_standard.npz", standard_answer)