import numpy as np
from ssk_cython import *

def weighted_degree_kxx(x, xp, d):
    assert len(x) == len(xp)
    l = len(x)
    sim = 0

    for k in range(1, d):
        beta_k = 2*(d-k+1)/float((d*(d+1)))
        sim_k = 0
        for i in range(1, l-k+1):
            if x[i:i+k] == xp[i:i+k]:
                sim_k += 1.
        sim += beta_k * sim_k

    return sim

def WD_K(sequences, d=4, cython=False):
    num_sequences = len(sequences)
    K = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(num_sequences):
            if cython:
                K[i, j] = cython_weighted_degree_kxx(sequences[i], sequences[j], d=d)
            else:
                K[i, j] = weighted_degree_kxx(sequences[i], sequences[j], d=d)

    return K

if __name__ == '__main__':
    x1 = 'ATCGATCG'
    x2 = 'ATCGATCG'
    x3 = 'ATCGATCC'
    x4 = 'ATCGATAA'
    x5 = 'ANNNNNNN'
    K = WD_K([x1, x2, x3, x4, x5])
