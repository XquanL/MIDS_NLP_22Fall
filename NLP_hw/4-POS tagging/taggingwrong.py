from itertools import count
import random
import re
import numpy as np
import nltk

# corpus 是一个tuple的list，每个tuple包含一个句子和一个tag的list [[('The', 'DET'), ('Fulton', 'NOUN'), ],[('The', 'DET'), ('Fulton', 'NOUN'), ]]
# HHM transition matrix
tags = []


def A(corpus, tags):
    for i in corpus:  # [('The', 'DET'), ('Fulton', 'NOUN'), ]
        for j in i:
            tags.append(j[1])
    tags = sorted(set(tags))
    transition_matrix = np.ones((len(tags), len(tags)))
    for i in corpus:
        for j in range(len(i) - 1):
            transition_matrix[tags.index(i[j][1])][tags.index(i[j + 1][1])] += 1
    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1, keepdims=True
    )
    return transition_matrix


# HHM emission matrix
words = []


def B(corpus, tags, words):
    for i in corpus:  # [('The', 'DET'), ('Fulton', 'NOUN'), ]
        for j in i:
            words.append(j[0])
    words = sorted(set(words))

    words.append("OOV/UNK")  # how to deal with OOV/UNK?

    emission_matrix = np.ones((len(tags), len(words)))
    for i in corpus:
        for j in i:
            emission_matrix[tags.index(j[1])][words.index(j[0])] += 1
    emission_matrix = emission_matrix / np.sum(emission_matrix, axis=1, keepdims=True)
    return emission_matrix

    # HHM initial state distribution
    # def pi(corpus, tags):
    # stat_dist = np.zeros(len(tags))
    # smoothing
    # stat_dist = np.ones(len(tags))
    # for i in range(len(tags)):
    # stat_dist[i] = 1 / len(tags)
    # return stat_dist

    # unigram mode?
    for i in corpus: # [('The', 'DET'), ('Fulton', 'NOUN'), ]
    unigram_model = {tag: corpus.count(tag) for tag in tags}
    stat_dist = np.array([unigram_model[tag] for tag in tags])
    stat_dist /= sum(stat_dist)
    return stat_dist


if __name__ == "__main__":
    corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    A = A(corpus, tags)
    B = B(corpus, tags, words)
# pi = pi(corpus, tags)
