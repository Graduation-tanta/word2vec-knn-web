from __future__ import print_function

import pickle
import numpy as np
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
from sklearn import decomposition
import random

with open('../pickleFiles/wikipedia.pic', 'rb') as f:
    myset = pickle.load(f)

reverse_dictionary = myset['reverse_dictionary']
dictionary = myset['dictionary']
count = myset['count']
final_embeddings = myset['final_embeddings']
del myset


# compares words similarity
def compareSim(word1, word2):
    x = np.matmul(word1, word2)
    return x


top_k = 8


# k nearest neighbours
def k_NN(word):
    nearest_words = []
    if word in dictionary:
        similarity = (np.matmul(final_embeddings[dictionary[word], :].reshape([-1, 1]).T, final_embeddings.T)).reshape([-1])
        nearest = (-similarity[:]).argsort()[1:top_k + 1]

        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            nearest_words.append(close_word)

    return nearest_words


# #test 0
# first num_points words
####################################################
# num_points = 50
# wordsInd = np.arange(1, num_points + 1)
# words = [reverse_dictionary[ele] for ele in wordsInd]
# Tsne_Embeddings = final_embeddings[wordsInd, :]
####################################################


# # random words #
# #test 1
####################################################
# num_points = 20
# wordsInd = np.array(random.sample(range(final_embeddings.shape[0]), num_points))
# words = [reverse_dictionary[ele] for ele in wordsInd]
# Tsne_Embeddings = final_embeddings[wordsInd, :]
####################################################



# # test 2
####################################################
# words = ['man', 'woman', 'men', 'women', 'uncle', 'aunt', 'uncles', 'aunts']
# wordsInd = [dictionary[s] for s in words]
# for i in range(0, len(words), 2):
#     print(compareSim(final_embeddings[dictionary[words[i]], :], final_embeddings[dictionary[words[i + 1]], :]))
#
# Tsne_Embeddings = final_embeddings[wordsInd, :]
####################################################

# # my words #
# # test 3
####################################################
testwords = ['machine', 'dynamic', 'surrounding',
             'four', 'cat', 'egg', 'abdallah', 'batman', 'blood', 'stream', 'intelligence', 'top', 'hitler', 'april', 'traffic',
             'khalil', 'king', 'cancer', 'league', 'book', 'nvidia', 'obama', 'story', 'mind', 'marvel', 'quiz',
             'sector', 'universe', 'kill', 'uncle', 'man', 'fantastic', 'usually', 'sunday', 'five', 'god', 'technology', 'soviet']

words = []  # to print
wordsInd = []  # for T-sne
shift = 1
Tsne_Embeddings = np.zeros(shape=(len(testwords) * (top_k + 1), final_embeddings.shape[1]))
for index, word in enumerate(testwords):
    nearest = k_NN(word)
    print('Nearest to %s:' % word, nearest)

# if nearest:
#         words.append(word)
#         words.extend(nearest)
#         wordsInd.clear()
#         wordsInd.append(dictionary[word])
#         for nearWord in nearest:
#             wordsInd.append(dictionary[nearWord])
#
#     # print(Tsne_Embeddings.shape)
#     # print(final_embeddings.shape)
#     #
#     # print(wordsInd[0: top_k + 1])
#     # print(index)
#     # print(Tsne_Embeddings[index * top_k: index * top_k + top_k + 1, :].shape)
#     # print(final_embeddings[wordsInd[0: top_k + 1], :].shape)
#
#     Tsne_Embeddings[index: index + top_k + 1, :] = final_embeddings[wordsInd[0: top_k + 1], :] * shift
#
#     shift *= 2

####################################################
with open('../pickleFiles/text8.pic', 'rb') as f:
    myset = pickle.load(f)

reverse_dictionary = myset['reverse_dictionary']
dictionary = myset['dictionary']
count = myset['count']
final_embeddings = myset['final_embeddings']
del myset

num_points = 300
wordsInd = np.arange(1, num_points + 1)
words = [reverse_dictionary[ele] for ele in wordsInd]
Tsne_Embeddings = final_embeddings[wordsInd, :]

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(Tsne_Embeddings)


# pca = decomposition.PCA(n_components=2)
# pca.fit(Tsne_Embeddings)
# two_d_embeddings = pca.transform(Tsne_Embeddings)


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    pylab.figure('Moahmmed alaa elkomy visualizer', figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


plot(two_d_embeddings, words)
