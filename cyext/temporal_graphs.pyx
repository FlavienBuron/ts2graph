# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
from lib.math cimport fmin

# Declare the decay function as a Python callback
ctypdef double (*decay_func_type)(int, int)

def k_hop_graph(np.ndarray[np.float32_t, ndim=2] x,
                int num_nodes=1,
                int k=1,
                bint bidirectional=True,
                object decay=None):
    """
    Cython version of k-hop temporal graph construction.
    Return edge_index and edge_weight as NumPy arrays
    """
    cdef int time_steps = x.shape[0]
    cdef int node, offset, src_idx
