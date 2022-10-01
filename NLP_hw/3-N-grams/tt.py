"""Markov Text Generator.

Patrick Wang, 2021

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""

import nltk

from mtg import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())

    words = finish_sentence(
        ["the", "interesting", "thing"],
        3,
        corpus,
        deterministic=True,
    )
    print(words)


if __name__ == "__main__":
    test_generator()
