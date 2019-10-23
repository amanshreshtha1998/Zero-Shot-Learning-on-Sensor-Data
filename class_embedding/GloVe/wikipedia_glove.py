# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:21:32 2018

@author: RAJDEEP PAL
"""

from gensim.scripts.glove2word2vec import glove2word2vec

glove_path = 'F:/year 2/hpg/project/attribute_embedding/GloVe/glove.6B.300d.txt'
w2v_path =   'F:/year 2/hpg/project/attribute_embedding/GloVe/glove.6B.300d.txt.word2vec'
glove2word2vec(glove_path, w2v_path)



#%%

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(w2v_path, binary = False)

results = model.most_similar(positive = ['woman', 'king'], negative = ['man'], topn = 3)
print (results)

#%%

words = list (model.vocab)
print (len(words))

print ( model['walk'] )
print ('brush_teeth' in words)

a = model['toothbrush']
b = model['teeth']
c = (a+b)/2
print (c)
#%%

X = model[model.vocab]

from sklearn.decomposition import PCA
from matplotlib import pyplot

pyplot.figure(figsize = (20, 20))
X = X[400:600]
temp = words[400:600]
# REDUCE DIMENSIONS
pca = PCA(n_components = 2)
results = pca.fit_transform(X)

# CREATE A SCATTER PLOT
pyplot.scatter(results[:, 0], results[:, 1])

# ANNOTATE POINTS
for i, word in enumerate(temp):
    pyplot.annotate(word, xy = (results[i, 0], results[i, 1]))
pyplot.show



























