from itertools import count
import random
import re
import numpy as np
import nltk
from collections import Counter
from viterbi import viterbi

# corpus： [[('The', 'DET'), ('Fulton', 'NOUN'), ],[('The', 'DET'), ('Fulton', 'NOUN'), ]]


def main(corpus):
    # HHM transition matrix
    tags = []
    for i in corpus:  # [('The', 'DET'), ('Fulton', 'NOUN'), ]
        for j in i:
            tags.append(j[1])
    tags = sorted(set(tags))
    # print(tags)
    transition_matrix = np.ones((len(tags), len(tags)))
    for i in corpus:
        for j in range(len(i) - 1):
            transition_matrix[tags.index(i[j][1])][tags.index(i[j + 1][1])] += 1
    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1, keepdims=True
    )
    # print(transition_matrix)

    # HHM emission matrix
    words = []
    for i in corpus:  # [('The', 'DET'), ('Fulton', 'NOUN'), ]
        for j in i:
            words.append(j[0])
    words = sorted(set(words))

    words.append("OOV/UNK")  # deal with OOV/UNK observation

    emission_matrix = np.ones((len(tags), len(words)))  # smoothing
    for i in corpus:
        for j in i:
            emission_matrix[tags.index(j[1])][words.index(j[0])] += 1
    emission_matrix = emission_matrix / np.sum(emission_matrix, axis=1, keepdims=True)
    # print(emission_matrix)

    # HHM initial state distribution
    # start of sentence like 'det' in corpus[0]
    stat_dist = np.zeros(len(tags))  # smoothing? np.ones(len(tags))?
    for i in corpus:
        stat_dist[tags.index(i[0][1])] += 1
    stat_dist = stat_dist / np.sum(stat_dist)
    # print(stat_dist)

    # test
    testset = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
    ct = 1
    total_error = 0
    for i in testset:
        print(f"Test case {ct}")
        obs = []
        for j in i:
            if j[0] in words:
                obs.append(words.index(j[0]))
            else:
                print("[OOV/UNK]:", j[0])
                obs.append(words.index("OOV/UNK"))
        # print(obs)
        v_result = viterbi(obs, stat_dist, transition_matrix, emission_matrix)
        # print(v_result)
        v_tag = [tags[x] for x in v_result[0]]
        print(f"Viterbi tag: {v_tag}")

        true_tag = [x[1] for x in i]
        print(f"True tag: {true_tag}")
        error = (np.array(v_tag) != np.array(true_tag)).sum()
        print(f"Error: {error}\n\n")
        total_error += error
        ct += 1
    print(f"Total error: {total_error}")


if __name__ == "__main__":
    corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    main(corpus)
