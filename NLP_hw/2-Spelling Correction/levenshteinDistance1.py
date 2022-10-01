'''A spelling corrector based on Levenshtein distance

IDS 703
Xiaoquan Liu   
Sept,2022'''

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
    def probability():
        total_words = sum(my_dict.values())
        my_dict_prob= {word: value/total_words for word, value in my_dict.items()}
        return my_dict_prob
    return probability()

my_dict_prob = get_data()


def calculate_distance_and_generate_list(word):
    wordlist = []
    distancelist = [levenshteinDistanceDP(word, key) for key in my_dict_prob.keys()]
    distance = min(distancelist)
    for key in my_dict_prob.keys():
        if levenshteinDistanceDP(word, key) == distance:
            wordlist.append(key)
    #print(wordlist)
    def spelling_corrector(word):
        if word in my_dict_prob.keys():
            return word
        else:
            if wordlist:
                return max(wordlist, key = my_dict_prob.get)
            else:
                print ('Could not find a correction for the word: {}'.format(word))

    return spelling_corrector

def spelling_corrector(word):
    return calculate_distance_and_generate_list(word)(word)


if __name__ == "__main__":
    get_data()


