import requests

FILE_URL = 'https://norvig.com/ngrams/count_1w.txt'

#create dictionary of words and the times they appear
my_dict = {}
def live_parse_data(file_url=FILE_URL):
    response = requests.get(file_url)
    if (response.status_code):
        data = response.text
        for i, line in enumerate(data.split('\n')):
            if '\t' in line:
                key = line.split('\t')[0]
                value = line.split('\t')[1]
                my_dict[key] =int(value)
                pass
            pass
        pass
    pass

#see the dictionary
#my_dict

#create probability dictionary of words
total_words = sum(my_dict.values())
my_dict_prob= {word: value/total_words for word, value in my_dict.items()}

#seeing the probability of the word 'of'
#my_dict_prob['of']


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


#the spelling corrector
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
            pass




if __name__ == "__main__":
    live_parse_data()



#good example "ehat" is corrected to "that"
#when a word needed to be corrected twice, this function will not work well
#if the correct word we need is not in the dictionary, the function will not work well