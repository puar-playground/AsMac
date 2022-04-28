#cython: boundscheck=False, cdivision=True, wraparound=False


import numpy as np
from libc.math cimport log
from libc.math cimport exp


cdef double softnw_score(double[:, :] seq1, double[:, :] seq2, double gap, double gamma):
    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L1 + 1, L2 + 1])
    cdef int i, j, d
    cdef double p_max = 0

    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        F[0, j] = -1 * j * gap
        for i in range(1, L1+1):
            for d in range(dim):
                if seq1[d, i-1] == 1:
                    p_max = F[i-1, j-1] + seq2[d, j-1]
                    break

            if (F[i-1, j] - gap) > p_max:
                p_max = (F[i-1, j] - gap)
            if (F[i, j-1] - gap) > p_max:
                p_max = (F[i, j-1] - gap)
            F[i, j] = p_max

    return F[L1, L2]


def best_hit_fast(double[:, :] seq, double[:, :] weight, double gap):

    cdef int L_r = weight.shape[1]
    cdef int L_c = seq.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L_r + 1, L_c + 1])
    cdef int i, j, d, j_max = 0
    cdef double p_max = 0, last_row_max = 0

    for i in range(1, L_r+1):
        F[i, 0] = -1 * i * gap
        for j in range(1, L_c+1):
            for d in range(dim):
                if seq[d, j-1] == 1:
                    p_max = F[i-1, j-1] + weight[d, i-1]
                    break

            if (F[i-1, j] - gap) > p_max:
                p_max = (F[i-1, j] - gap)
            if (F[i, j-1] - gap) > p_max:
                p_max = (F[i, j-1] - gap)

            F[i, j] = p_max

    for j in range(1, L_c+1):
        if F[L_r, j] > last_row_max:
            j_max = j
            last_row_max = F[L_r, j]

    i = L_r
    j = j_max
    cdef double diag
    while i != 0 and j != 0:
        for d in range(dim):
            if seq[d, j-1] == 1:
                diag = F[i-1, j-1] + weight[d, i-1]
                break
        if diag == F[i, j]:
            i -= 1
            j -= 1
        elif (F[i-1, j] - gap) == F[i, j]:
            i -= 1
        elif (F[i, j-1] - gap) == F[i, j]:
            j -= 1
        else:
            print(diag)
            print("Wrong! in trace back")

    return last_row_max, j, j_max


def embed_value_fast(double[:, :] seq, double[:, :] weight, double gap):

    cdef int L_r = weight.shape[1]
    cdef int L_c = seq.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L_r + 1, L_c + 1])
    cdef int i, j, d, j_max = 0
    cdef double p_max = 0, last_row_max = 0

    for i in range(1, L_r+1):
        F[i, 0] = -1 * i * gap
        for j in range(1, L_c+1):
            for d in range(dim):
                if seq[d, j-1] == 1:
                    p_max = F[i-1, j-1] + weight[d, i-1]
                    break

            if (F[i-1, j] - gap) > p_max:
                p_max = (F[i-1, j] - gap)
            if (F[i, j-1] - gap) > p_max:
                p_max = (F[i, j-1] - gap)

            F[i, j] = p_max

    for j in range(1, L_c+1):
        if F[L_r, j] > last_row_max:
            j_max = j
            last_row_max = F[L_r, j]

    return last_row_max



def softnw_f_fast(double[:, :] seq1, double[:, :] seq2, double gap, double gamma):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef double[:, :] F = np.zeros([L1 + 2, L2 + 2])
    cdef double a=0, b=0, c=0, assign_temp = 0
    cdef int i, j, d
    cdef int dim = 4

    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        F[0, j] = -1 * j * gap
        for i in range(1, L1+1):
            for d in range(dim):
                if seq1[d, i-1] == 1:
                    a = (F[i-1, j-1] + seq2[d, j-1])/gamma
                    break

            b = (F[i-1, j] - gap)/gamma
            c = (F[i, j-1] - gap)/gamma
            p_max = a
            if b > p_max:
                p_max = b
            if c > p_max:
                p_max = c

            assign_temp += exp(a - p_max)
            assign_temp += exp(b - p_max)
            assign_temp += exp(c - p_max)
            assign_temp = gamma * (log(assign_temp) + p_max)
            F[i, j] = assign_temp
            assign_temp = 0
    return F


def softnw_q_fast(double[:, :] seq1, double[:, :] seq2, double[:, :] F, double gap, double gamma):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef int i, j, d, dim = 4
    cdef double a, b, c, dot_product = 0
    cdef double[:, :] Q = np.zeros([L1 + 2, L2 + 2])

    F[L1+1, L2+1] = F[L1, L2]
    Q[L1+1, L2+1] = 1

    for i in range(1, L1+1):
        F[i, L2+1] = 1e8

    for j in [L2 - x for x in range(L2)]:
        F[L1+1, j] = 1e8
        for i in [L1 - x for x in range(L1)]:
            if i == L1 or j == L2:
                dot_product = 0
            else:
                for d in range(dim):
                    if seq1[d, i] == 1:
                        dot_product = seq2[d, j]
                        break

            if F[i, j] + dot_product - F[i+1, j+1] > 0:
                a = 1
            else:
                a = exp((F[i, j] + dot_product - F[i+1, j+1])/gamma)

            b = exp((F[i, j] - gap - F[i + 1, j]) / gamma)
            c = exp((F[i, j] - gap - F[i, j + 1]) / gamma)

            Q[i, j] = a * Q[i+1, j+1] + b * Q[i+1, j] + c * Q[i, j+1]

    return Q


def softnw_p_fast(double[:, :] seq1, double[:, :] seq2, double[:, :] F, double gap, double gamma):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef double[:, :] P = np.zeros([L1 + 2, L2 + 2])
    cdef double a=0, b=0, c=0
    cdef int i, j, d
    cdef int dim = 4

    for i in range(L1+1):
        P[i, 0] = -1 * i
    for j in range(1, L2+1):
        P[0, j] = -1 * j
        for i in range(1, L1+1):

            for d in range(dim):
                if seq1[d, i-1] == 1:
                    dot_product = seq2[d, j-1]
                    break

            a = exp((F[i-1, j-1] + dot_product - F[i, j])/gamma)
            b = exp((F[i-1, j] - gap - F[i, j]) / gamma)
            c = exp((F[i, j-1] - gap - F[i, j]) / gamma)

            P[i, j] = a * P[i-1, j-1] + b * (P[i-1, j]-1) + c * (P[i, j-1]-1)

    return P[L1, L2]


def softnw_h_fast(double[:, :] seq1, double[:, :] seq2, double[:, :] F):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef double[:, :] H = np.zeros([L1+1, L2+1])
    cdef int i, j, d, dim = 4

    for j in range(1, L2+1):
        for i in range(1, L1+1):
            for d in range(dim):
                if seq1[d, i-1] == 1:
                    H[i, j] = F[i-1, j-1] + seq2[d, j-1] - F[i, j]
                    break
    return H


def seq2_gradient_fast(double[:, :] seq1, double[:, :] seq2, double[:, :] Q, double[:, :] H, double gamma):

    cdef double[:, :] s2_g = np.zeros([seq2.shape[0], seq2.shape[1]])
    cdef int i, j, d, dim = 4
    cdef double p
    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]

    for j in range(1, L2+1):
        for i in range(1, L1+1):
            p = Q[i, j] * exp(H[i, j] / gamma)
            for d in range(dim):
                s2_g[d, j-1] += p * seq1[d, i-1]

    return s2_g
