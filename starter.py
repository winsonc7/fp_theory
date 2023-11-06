import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye

A = 2       # forward rate
B = 1       # backward rate
N = 5       # matrix size
T = 1       # time
K = 10      # num terms to sum


def generate_rate_matrix(a, b, n):
    rate_matrix = csr_matrix((n,n), dtype=float)
    for i in range(n):
        if i + 1 < n - 1:   # bounds checking
            rate_matrix[i][i+1] = a
        if i != 0:          # bounds checking
            rate_matrix[i][i-1] = b
        rate_matrix[i][i] = sum(rate_matrix[i]) * -1
    return rate_matrix

def factorial_memo(n):
    memo = [1]
    for i in range(1, n):
        memo.append(memo[-1] * i)
    print("done")
    return memo

def standard_exponential(q, t, k):
    f_memo = factorial_memo(k)      # calculates factorials beforehand, idk felt like it'd be faster
    new_q = t * q
    curr_q = eye(new_q.shape[0], format='csr')          # identity matrix
    sum = curr_q
    for i in range(1, k):
        curr_q = curr_q @ new_q         # multiplication of sparse matrices
        sum += new_q / f_memo[i]
        print(i)
    return sum

# q_matrix = generate_rate_matrix(A, B, N)
# print("done")
# standard_answer = standard_exponential(q_matrix, T, K)
# print(standard_answer)