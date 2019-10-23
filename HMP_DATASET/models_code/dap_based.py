# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:09:04 2018

@author: RAJDEEP PAL
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools

n_cls = 14
all_cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
n_att = 15
seed = 0
#%% LOAD DATASET



directory = 'F:/year 3/zsl/HMP_DATASET/extracted_features/zero_to_nine'
arr = os.listdir(directory)

feature_names = ['maxX', 'minX', 'avgX', 'stdX', 'slopeX', 'zcrX',     'maxY','minY','avgY','stdY', 'slopeY', 'zcrY',          'maxZ', 'minZ', 'avgZ', 'stdZ', 'slopeZ', 'zcrZ',      'maxACC', 'minACC', 'avgACC', 'stdACC',     'XYcorr', 'YZcorr','ZXcorr',    'energy']

path = directory+'/'+arr[0]
data = []

for file_name in arr:
    path = directory + '/' + file_name
    df  = pd.read_csv(path, names = feature_names)
    data.append(df)
    print (df.shape)


directory = 'F:/year 3/zsl/HMP_DATASET/extracted_features/ten_to_thirteen'
arr = os.listdir(directory)

for file_name in arr:
    path = directory + '/' + file_name
    df  = pd.read_csv(path, names = feature_names)
    data.append(df)
    print (df.shape)


#%% LOAD ACTIVITY ATTRIBUTE MATRIX
    
 # F:\year 2\hpg\project\activity_attribute_matrix.csv
att_names = ['sittting', 'standing', 'walking', 'posture_upright', 'wrist_movement', 'arm_pendulum_swing', 'hands_on_table', 'hand_above_chest', 'translation_motion', 'cyclic_motion', 'meal_related', 'morning', 'evening', 'potential_energy_increase', 'potential_energy_decrease']
aam = pd.read_csv('F:/year 3/zsl/HMP_DATASET/models_code/matrix.csv', names = att_names)
print (aam)


class_names = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk'] 



#%%
def get_clf(attribute, ts_cls):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC, OneClassSVM
    
    ex = pd.DataFrame()
    tr_cls = [x for x in all_cls if (x not in ts_cls)]
    
    
    for cls in tr_cls:
        #print ('class', cls)
        df = data[cls]
        m, n = df.shape
        # print (m)
        
        if (attribute[cls] == 1):
            tgt_df = pd.DataFrame(np.ones(m), columns = ['target'])
            #print (1)
        else:
            tgt_df = pd.DataFrame(np.zeros(m), columns = ['target'])
            #print (0)
            
        df = df.join(tgt_df)
        ex = ex.append(df, ignore_index = True)
        
    X = ex[feature_names]
    y = ex['target']
    # print ('abc', y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    if (y[y == 1].size == 0 or y[y == 0].size == 0):
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, random_state = seed).fit(X_train)
        predict = clf.predict(X_test)
        acc = accuracy_score(y_test, predict)
    else:
        clf = SVC(random_state = seed).fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

    return (clf, acc)
    
        


#%%



def train_clf(ts_cls, att_acc):
    clf_arr = []
    for i, attribute in enumerate(att_names):
        attribute_vector = aam[attribute]
        (clf, acc) = get_clf(attribute_vector, ts_cls)
        
        clf_arr.append(clf)
        att_acc[i] += acc 
    return clf_arr


#%%
        
def get_predicted_attribute_vec(x, clf_arr):
    y_pred = []
    for attribute in range(0, n_att):
        clf = clf_arr[attribute]
        predicted_attribute = clf.predict(x)
        
        y_pred.append(predicted_attribute)
    
    return y_pred
#%%

# TEST FOR A INPUT FEATURE FROM CLASS 1

def most_likely_class(x_test, ts_cls, clf_arr):
    
   from sklearn.metrics import accuracy_score
    
   x = x_test
   score = []
   for cls in ts_cls:
        
        y_pred = get_predicted_attribute_vec(x, clf_arr)
        y_test = aam[cls:cls+1]
        y_test = pd.DataFrame.transpose(y_test)
        # print (y_test.shape, len(y_pred))
        similarity = accuracy_score(y_test, y_pred)
        score.append(similarity)
   if score[0] > score[1]:
       return ts_cls[0]
   else:
       return ts_cls[1]
   
    
    



#%%

lt = ['abc', 'def']
for i, st in enumerate(lt):
    print (i, st)

#%%
def evaluate(X, clf_arr, ts_cls, true_cls):
    
    
    
    (m, n) = X.shape
    #y_true = np.ones(m)
    y_pred = np.zeros(m)
    for row in range(0, m):
        x_test = X[row:row+1]
        predicted_cls = most_likely_class(x_test, ts_cls, clf_arr)
#        if predicted_cls == true_cls:
#            y_pred[row] = 1
        y_pred[row] = predicted_cls
        
    #a = accuracy_score(y_true, y_pred)
    #p = average_precision_score(y_true, y_pred)
    #r = recall_score(y_true, y_pred)
    #p = 0
    #r = 0
    #print (a, p, r)
    return y_pred
        
        

#%%  LEAVE 2 CLASS OUT CROSS VALIDATION
    
a = 0
p = 0
r = 0
f1 = 0

att_acc = np.zeros(n_att)

count = 0


for i in range(0, n_cls):
    X_test_i = data[i]
    mi = X_test_i.shape[0]
    for j in range(i+1, n_cls):
        
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        count += 1
        ts_cls = [i, j]
        clf_arr = train_clf(ts_cls, att_acc)
        
        
        pred = evaluate(X_test_i, clf_arr, ts_cls, i)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))

        
        X_test_j = data[j]
        pred = evaluate(X_test_j, clf_arr, ts_cls, j)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.ones(pred.shape, dtype = int))
        
        y_pred[y_pred==i] = 0
        y_pred[y_pred==j] = 1
        
        a += accuracy_score(y_true, y_pred)
        p += precision_score(y_true, y_pred)
        r += recall_score(y_true, y_pred)
        f1 += f1_score(y_true, y_pred)
        
    
        
print (count)
#accuracy = accuracy / (n_cls - 1)
#att_acc = att_acc / count
#precision = precision / count
#recall = recall / count      
    
print ( a/count, p/count, r/count, f1/count )



#print (accuracy)
#print (accuracy.mean(axis = 0) * 100)
#print (att_acc)




#%%

print (y_true.shape, y_pred.shape, accuracy_score(y_true, y_pred))
print (classification_report(y_true, y_pred, target_names = class_names))
cnf = confusion_matrix(y_true, y_pred)
print (cnf)



#%%
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.figure(figsize = (15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



np.set_printoptions(precision=1) 
fig, ax = plt.subplots()
plot_confusion_matrix(cnf)


#%%
from gensim.models import Word2Vec

# DEFINE TRAINING DATA
wiki = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
        
 


# TRAIN MODEL
model = Word2Vec(wiki, min_count = 1)

print (model)
#vocab = number of distinct words which has appeared at least min_count number of times




# SUMMARIZE VOCABLUARY
words = list (model.wv.vocab) # list of all distinct words in vocab
print (words)




# ACCESS WORD VECTOR FOR ONE WORD
print (model['sentence'])



# SAVE THE MODAL
model.save('model.bin')



# LOAD MODEL
new_model = Word2Vec.load('model.bin')
print (new_model)





























