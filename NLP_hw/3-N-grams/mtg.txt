"""Markov text generator
n-gram model

Xiaoquan Liu, 2022
"""

import random
import re
import numpy as np
import nltk


def finish_sentence(sentence, n, corpus, deterministic):
    # corpus is a list of words -- ['[', 'emma', 'by', 'jane', 'austen', '1816', ']', ...]
    # token_index_dict = Counter(corpus)
    distinct_tokens = list(set(corpus))
    distinct_tokens_count = len(distinct_tokens)
    token_index_dict = {token: i for i, token in enumerate(distinct_tokens)}

    sets_of_n_words = [
        " ".join(corpus[i : i + (n - 1)]) for i, _ in enumerate(corpus[: -(n - 1)])
    ]  # n-gram list

    # n_gram_dict = Counter(sets_of_n_words)  # n-gram dict
    distinct_n_grams = list(set(sets_of_n_words))
    distinct_n_grams_count = len(distinct_n_grams)
    n_gram_dict = {n_gram: i for i, n_gram in enumerate(distinct_n_grams)}

    n_gram_matrix = np.zeros((distinct_n_grams_count, distinct_tokens_count))

    for i, word in enumerate(sets_of_n_words[: -(n - 1)]):
        word_sequence_index = n_gram_dict[word]
        next_word_index = token_index_dict[corpus[i + (n - 1)]]
        n_gram_matrix[word_sequence_index, next_word_index] += 1

    # normalize the matrix
    np.seterr(divide="ignore", invalid="ignore")
    n_gram_matrix = n_gram_matrix / n_gram_matrix.sum(axis=1, keepdims=True)

    while len(sentence) < 10:
        current_sentence = " ".join(sentence[-(n - 1) :])
        dist = n_gram_matrix[n_gram_dict[current_sentence]]
        if deterministic:
            next_word_index = dist.argmax()
            next_word = distinct_tokens[next_word_index]
            sentence.append(next_word)
        else:
            next_word = random.choices(distinct_tokens, dist)
            sentence.append(next_word[0])
        if next_word in [".", "!", "?"]:
            break

    return sentence