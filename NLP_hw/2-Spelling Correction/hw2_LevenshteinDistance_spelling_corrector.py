'''A spelling corrector based on Levenshtein distance

IDS 703
Xiaoquan Liu   
Sept,2022'''

import numpy as np

def levenshteinDistanceDP(word1, word2):
    m = len(word1)
    n = len(word2)
    mtx = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                mtx[i][j] = j
            elif j == 0:
                mtx[i][j] = i
            elif word1[i-1] == word2[j-1]:
                mtx[i][j] = mtx[i-1][j-1]
            else:
                mtx[i][j] = 1 + min(mtx[i][j-1], mtx[i-1][j], mtx[i-1][j-1])
    return mtx[m][n]

my_dict = {}
def get_data():
    with open("count_1w.txt") as f:
        for line in f:
            key = line.split()[0]
            value = line.split()[1]
            my_dict[key] = int(value)
        return my_dict

def mindistance_word(word):
    p = 0.00001
    max_now = -10**1000000
    total_words = sum(my_dict.values())
    for potential_word in my_dict.keys():
        E = levenshteinDistanceDP(word, potential_word)
        w = E * np.log(p) + np.log(my_dict[potential_word]/total_words)
        if w > max_now:
            max_now = w
            max_word = potential_word
    return max_word


def spelling_corrector(word):
    if word in my_dict.keys():
        return word
    else:
        return mindistance_word(word)


if __name__ == "__main__":
    get_data()