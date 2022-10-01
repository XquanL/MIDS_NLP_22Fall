"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
import numpy as np

# 对于每一条 sentence 每一个词找到index
# 有一些词不在词典里，用OOV/UNK代替
# obs是一个list，里面是每个词的index

# 最后还要计算你算出来的数据有多少是正确的

# infer the sequence of states for sentences 10150-10152 of the Brown corpus


def viterbi(obs, pi, A, B):
    """Viterbi POS tagging."""
    n = len(obs)

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]  # for t = 0, starts with those pis
    # np.log(B[:, obs[0]]) is the emission probability of the first word (column of B associated with the first observation)
    # prob for each  states = the prob for the observation given each state multiplied by initial state prob
    log_psi = [0]

    # recursion
    for z in obs[1:]:  # walk through for each observation
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(
            A
        )  # be in state s/s/r/r given state r/s/s/t (all four combinations)
        log_d.append(
            np.max(log_da, axis=0) + np.log(B[:, z])
        )  # max prob of being in each state given the observation at this pt
        log_psi.append(np.argmax(log_da, axis=0))  # give the backward pointer(argmax)

    # termination
    log_ps = np.max(log_d[-1])
    qs = [np.empty((0,))] * n
    qs[-1] = np.argmax(log_d[-1])  # final state = state with highest prob
    for i in range(n - 2, -1, -1):  # walk backwards through the backword pointers
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)  # state sequence and probability of the sequence
