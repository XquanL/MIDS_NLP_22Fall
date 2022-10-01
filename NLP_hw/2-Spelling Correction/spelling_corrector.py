'''A spelling corrector based on Levenshtein distance

IDS 703
Xiaoquan Liu   
Sept,2022'''

my_dict = {}
def get_data():
    with open("count_1w.txt") as f:
        for line in f:
            key = line.split()[0]
            value = line.split()[1]
            my_dict[key] = int(value)
        return my_dict


#spliting the word into two parts
def split(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


#editing operations for Levenshtein distance
def insertion(word):
    return [left + insrt + right for left, right in split(word) for insrt in 'abcdefghijklmnopqrstuvwxyz']

def deletion(word):
    return [left + right[1:] for left, right in split(word) if right]

def substitution(word):
    return [left + sub + right[1:] for left, right in split(word) if right for sub in 'abcdefghijklmnopqrstuvwxyz']


#combine all the editing operations
def edit(word):
    return set(insertion(word) + deletion(word) + substitution(word))



#create probability dictionary of words
#contains the spelling corrector
def probability():
    total_words = sum(my_dict.values())
    my_dict_prob= {word: value/total_words for word, value in my_dict.items()}
    def spelling_corrector(word):
        if word in my_dict_prob.keys():
            return word
        else:
            candidates = edit(word)
            candidates = [word for word in candidates if word in my_dict_prob.keys()]
            if candidates:
                return max(candidates, key = my_dict_prob.get)
            else:
                print ('Could not find a correction for the word: {}'.format(word))
    return spelling_corrector

#the spelling corrector
def spelling_corrector(word):
    return probability()(word)



if __name__ == "__main__":
    get_data()
    probability()



#good example "ehat" is corrected to "that"
#when a word needed to be corrected twice, this function will not work well
#if the correct word we need is not in the dictionary, the function will not work well
#Levenshtein distance is not the best way to correct spelling---do not include transposition