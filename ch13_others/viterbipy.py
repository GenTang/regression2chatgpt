# -*- coding: UTF-8 -*-
'''
This script implements the Viterbi algorithm
'''


import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


def viterbi(obs, init_prob, trans_prob, emit_prob):
    '''
    Viterbi algorithm

    Parameters
    ----
    obs : {np.array or scipy.sparse.csr_matrix}, shape (number of samples, number of features),
        Feature matrix of the data

    initProb : {np.array or scipy.sparse.csr_matrix}, shape (number of states),
        Initial distribution over states

    transProb : {np.array or scipy.sparse.csr_matrix}, shape (number of states, number of states),
        State-transition matrix

    emitProb : {np.array or scipy.sparse.csr_matrix}, shape (number of states, number of features),
        Conditional feature probabilities for each state

    Returns
    ----
    score : {np.array}, shape (number of samples, number of states), intermediate probabilities in the Viterbi algorithm

    path : {np.array}, shape (number of samples), final hidden state for each sample
    '''
    sample_num, state_num = obs.shape[0], init_prob.shape[0]
    backp = np.empty((sample_num, state_num), dtype=np.intp)
    score = safe_sparse_dot(obs, emit_prob.T)

    for i in range(state_num):
        score[0, i] += init_prob[i]

    for i in range(1, sample_num):
        for j in range(state_num):
            max_ind = 0
            max_val = -np.inf
            for k in range(state_num):
                candidate = score[i - 1, k] + trans_prob[k, j] + score[i, j]
                if candidate > max_val:
                    max_ind = k
                    max_val = candidate
            score[i, j] = max_val
            backp[i, j] = max_ind

    path = np.empty(sample_num, dtype=np.intp)
    path[sample_num - 1] = score[sample_num - 1, :].argmax()
    for i in range(sample_num - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]
    return score, path