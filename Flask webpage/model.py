from __future__ import print_function
import pickle
import numpy as np

reverse_dictionary = [0, 0]
dictionary = [0, 0]
final_embeddings = [0, 0]

with open('../../pickleFiles/wikipedia.pic', 'rb') as f:
    myset = pickle.load(f)

reverse_dictionary[0] = myset['reverse_dictionary']
dictionary[0] = myset['dictionary']
final_embeddings[0] = myset['final_embeddings']
del myset

with open('../../pickleFiles/text8.pic', 'rb') as f:
    myset = pickle.load(f)

reverse_dictionary[1] = myset['reverse_dictionary']
dictionary[1] = myset['dictionary']
final_embeddings[1] = myset['final_embeddings']
del myset

top_k = 8


# k nearest neighbours if possible
def k_NN(word, model):
    nearest_words = []
    if word in dictionary[model]:
        similarity = (np.matmul(final_embeddings[model][dictionary[model][word], :].reshape([-1, 1]).T, final_embeddings[model].T)).reshape([-1])
        nearest = (-similarity[:]).argsort()[1:top_k + 1]

        for k in range(top_k):
            close_word = reverse_dictionary[model][nearest[k]]
            nearest_words.append(close_word)

    if nearest_words == []:
        nearest_words = ['404', 'معلش', '404', 'معلش', 'معلش', 'not found', 'مش لاقى', 'مهواش موجود', 'معرفشى', 'معلش', '404', 'معلش']
    return nearest_words
