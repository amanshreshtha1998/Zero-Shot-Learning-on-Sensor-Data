# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:41:38 2018

@author: RAJDEEP PAL
"""

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
#%%

glove_path = 'F:/year 3/zsl/class_embedding/GloVe/glove.6B.300d.txt'
w2v_path =   'F:/year 3/zsl/class_embedding/GloVe/glove.6B.300d.txt.word2vec'
glove2word2vec(glove_path, w2v_path)

#%%


model = KeyedVectors.load_word2vec_format(w2v_path, binary = False)

results = model.most_similar(positive = ['woman', 'king'], negative = ['man'], topn = 3)
print (results)


words = list (model.vocab)
print (len(words))



#%%

X = model[model.vocab]  # 400000 * 50
print (X.shape)



pyplot.figure(figsize = (20, 20))
X = X[900:1000]       # 200 * 50
temp = words[900:1000]
# REDUCE DIMENSIONS
pca = PCA(n_components = 2)
results = pca.fit_transform(X) # numpy array     # 200 * 2

# CREATE A SCATTER PLOT
pyplot.scatter(results[:, 0], results[:, 1])

# ANNOTATE POINTS
for i, word in enumerate(temp):
    pyplot.annotate(word, xy = (results[i, 0], results[i, 1]))
pyplot.show

#%%

#CLASS0  =  BRUSH_TEETH 



aam = pd.DataFrame()

a = model['brush']
b = model['toothbrush']
c = model['teeth']
c0 = (a + b + c ) / 3

c0 = pd.DataFrame(c0)             # data frame = 50 * 1
c0 = pd.DataFrame.transpose(c0)   # data frame = 1 * 50

aam = aam.append(c0, ignore_index = True)  # data frame = 1 * 50
print (aam.shape)





# CLASS 1

a = model['climb']
b = model['stairs']
d = 0.66*a + 0.33*b
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)





# CLASS 2
a = model['comb']
b = model['hair']
d = (a+b)/2
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)




# CLASS 3

a = model['descend']
b = model['stairs']
d = 0.66*a+0.33*b
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)




# CLASS 4

a = model['drink']
b = model['water']
d = (0.66*a+0.33*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape) 




# CLASS 5

a = model['eat']
b = model['meat']
d = (0.33*a+0.66*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)





# CLASS 6

a = model['eat']
b = model['soup']
d = (0.33*a+0.66*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)






# CLASS 7

print ('getup' in words)

a = model['getup']
b = model['bed']
d = (0.66*a+0.33*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)





# CLASS 8

print ('liedown' in words)

a = model['lie']
b = model['down']
c = model['bed']
d = (0.66*(a+b)+0.33*c)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)





# CLASS 9

print ('waterpour' in words)

a = model['pour']
b = model['water']
d = (0.66*a+0.33*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)





# CLASS 10

print ('sitdown' in words)

a = model['sitdown']
b = model['chair']
d = (0.66*a+0.33*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)




# CLASS 11

print ('standup' in words)

a = model['standup']
b = model['chair']
d = (0.66*a+0.33*b)
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)




# CLASS 12

print ('talktelephone' in words)

a = model['talk']
b = model['telephone']
d = (a+b)/2
d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)
print (aam.shape)



# CLASS 13

d = model['walk']

d = pd.DataFrame(d)
d = pd.DataFrame.transpose(d)

aam = aam.append(d, ignore_index  = True)  # data frame = 14 * 300
print (aam)

class_names = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']

#%%
  """ HMP """

#df = pd.read_csv('F:/year 3/zsl/class_embedding/activity_attribute_matrix300.csv')
class_names = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']
df = pd.DataFrame.transpose(aam)


  """ HAR """
#df = pd.read_csv('F:/year 3/zsl/HAR_DATASET/activity_attribute_matrix300.csv')
#class_names = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying']


df = pd.DataFrame.transpose(df)  
pca = PCA(n_components = 2)
results = pca.fit_transform(df)


pyplot.figure(figsize = (10, 10))
pyplot.scatter(results[:, 0], results[:, 1], marker = 'o')

for i, word in enumerate(class_names):
    pyplot.annotate(word, xy = (results[i, 0], results[i, 1]) )


pyplot.show


#%% SAVE ATTRIBUTE MATRIX NOT PROBABILITY IN CSV FILE
aam = aam.T
print (aam.shape)
aam.to_csv(r'F:/year 3/zsl/class_embedding/weighted/activity_attribute_matrix_not_probability300.csv', header = class_names, index = None)

#%% LOAD DATA

df = pd.read_csv('F:/year 3/zsl/class_embedding/weighted/activity_attribute_matrix_not_probability300.csv')
print (df, df.shape) # n_att * n_cls

#%% CONVERT DATAFRAME TO NUMPY ARRAY

z = np.array(df) # n_att * n_cls
print (z.shape)
#re = z[:, 11].reshape(14, 1)
#print (re.shape) # 1n_att * n_cls


#%%
def softmax(x):
    # Calculates the softmax for each row of the input x.
    # Argument: x -- A numpy matrix of shape (n_att, n_cls)

    # Returns: s -- A numpy matrix equal to the softmax of x, of shape (n_att, n_cls)
    
    

    # Apply exp() element-wise to x. 
    x_exp = np.exp(x)
    # Create a vector x_sum that sums each column of x_exp.
    x_sum = x_exp.sum(axis = 0, keepdims = True)
    print (x_sum.shape) # 1 * n_cls
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    print(s.shape) # n_att * n_cls
    
    
    return s


#%%

aam_p = softmax(z) # n_att * n_cls .  SUM ALONG COLUMNS IS 1
print (aam_p, aam_p.shape) 

x_sum = aam_p.sum(axis = 0, keepdims = True)
print (x_sum, x_sum.shape) 


#%% SAVE ATTRIBUTE MATRIX  PROBABILITY IN CSV FILE

aam_p_df = pd.DataFrame(aam_p) 
aam_p_df.to_csv(r'F:/year 3/zsl/class_embedding/weighted/activity_attribute_matrix300.csv', header = class_names, index = None)


#%% LOAD

df = pd.read_csv('F:/year 3/zsl/class_embedding/weighted/activity_attribute_matrix300.csv')
print (df, df.shape)


