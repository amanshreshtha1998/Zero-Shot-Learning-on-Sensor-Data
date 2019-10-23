# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:09:04 2018

@author: RAJDEEP PAL
"""

import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

n_cls = 6
all_cls = [0, 1, 2, 3, 4, 5]
n_att = 300
n_feat = 25
seed = 0

def l2_norm(f):
    f /= (np.linalg.norm(f) + 10e-6)
    return f


#%% LOAD DATASET

directory = 'F:/year 3/zsl/HAR_DATASET/extracted_features/final_extracted_features_32'
arr = os.listdir(directory)

#feature_names = ['maxX', 'minX', 'avgX', 'stdX', 'slopeX', 'zcrX',     'maxY','minY','avgY','stdY', 'slopeY', 'zcrY',          'maxZ', 'minZ', 'avgZ', 'stdZ', 'slopeZ', 'zcrZ',      'maxACC', 'minACC', 'avgACC', 'stdACC',     'XYcorr', 'YZcorr','ZXcorr',    'energy']
#feature_names = ['0', '1', '2', '3', '4', '5',     '6','7','8','9', '10', '11',          '13', '14', '15', '16', '17', '18',      '19', '20', '21', '22',     '23', '24','25',    '26', '27', '28', '29', '30', '31']
feature_names = list(range(0, n_feat))
path = directory+'/'+arr[0]
data = []


for file_name in arr:
    path = directory + '/' + file_name
    df  = pd.read_csv(path, index_col = False, names = feature_names)
    data.append(df)
    print (df.shape)

print (data[0])


#%%  ACTIVITY ATTRIBUTE MATRIX - GLOVE
    
 # F:\year 2\hpg\project\activity_attribute_matrix.csv

class_names = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying']

aam = pd.read_csv('F:/year 3/zsl/HAR_DATASET/activity_attribute_matrix300.csv')
print (aam.shape)


#%%

def get_max_rank_cls(x, S, W, true_label, tr_cls):
    x = x.reshape((1, n_feat))
    max_rank = -1
    max_rank_cls = -1
    #print (S[class_names[true_label]])
    for j in tr_cls:
        delta = 0 if (true_label == j) else 1
        proj = np.matmul(x, W)
        proj = l2_norm(proj).reshape((1, n_att))
        comp = 0
        comp = np.matmul(proj, S[class_names[true_label]].reshape((n_att, 1)) )
        curr_rank = delta + comp
        if curr_rank > max_rank:
            max_rank = curr_rank
            max_rank_cls = j
        
    
    return max_rank_cls
    
    

#%%
def get_mapping(ts_cls):
    
    
    
    X = pd.DataFrame()
    S = pd.DataFrame()
    labels = pd.DataFrame()
    #print (ts_cls)
    tr_cls = [x for x in all_cls if (x not in ts_cls)]
    #print (tr_cls)
    #print (tr_cls)
    
    
    for i, cls in enumerate(tr_cls):
        #print ('class', cls)
        df = data[cls]
        attribute_vec = aam[class_names[cls]]
        m1, d = df.shape
        
        y = np.ones((m1, 1), int) * int(cls)
        #y[:, i] = 1
        y = pd.DataFrame(y)
        #Y = Y.append(y_df, ignore_index = True)
        
        # print (m)
        S = S.join(attribute_vec, how = 'right')
        X = X.append(df, ignore_index = True)
        labels = labels.append(y, ignore_index = True)
    
    X = np.array(X)
    S = np.array(S)
    labels = np.array(labels)
    
    (m, d) = X.shape
    (a, z) = S.shape
    #print (m, d)
    #print (a, z)
    #print (labels.shape)
    
    
    n_train = X.shape[0]
    #print (n_train)
    #n_class = S.shape[1]
    #print (X[1].shape, S ,labels.shape)
    
    
    
    T = 100
    D = n_feat
    E = n_att
    W = np.ones((D, E))
    N = n_train
    alpha = 0.05
    indices = list(range(0, N))
    
    for t in range(0, T):
        
        random.shuffle(indices)
        for index in range(0, N):
            n = indices[index]
            x = X[n].reshape((1, n_feat))
            true_label = int (labels[n])
            #print (true_label)
            #s = S[true_label]
            y = get_max_rank_cls(x, aam, W, true_label, tr_cls)
            if ( (y != true_label) and (y != -1) ):
                diff = np.subtract(aam[class_names[true_label]], aam[class_names[y]]).T.reshape((1, E))
                x = x.T.reshape((D, 1))
                grad = np.matmul(x, diff).reshape((D, E))
                W = W + alpha * grad
                
    return (W)
                
            
    
#%%    
def most_likely_class(m, X, W, S):
    X = np.array(X)
    W = np.array(W)
    S = np.array(S)
    #print (X.shape, W.shape, S.shape)
    pred_att = np.matmul( X.reshape((m, n_feat)) , W.reshape((n_feat, n_att)) ).reshape((m, n_att))
    #print (pred_att.shape)
    dot_matrix = cosine_similarity(pred_att, S.T)
    #print (dot_matrix.shape)
    predicted_cls = np.argmax(dot_matrix, axis = 1).reshape((m, 1))
    #print (predicted_cls.shape)
    return predicted_cls
#%%
    
def evaluate(X_test, S, W, true_cls):
    
    (m, d) = X_test.shape
    y_pred = most_likely_class(m, X_test, W, S)
    return y_pred

    if true_cls == 0:
        y_true = np.zeros(y_pred.shape, dtype = int)
        return accuracy_score(y_true, y_pred)
    else:
        y_true = np.ones( y_pred.shape, dtype = int)
        return accuracy_score(y_true, y_pred)


#%%#%%  LEAVE 2 CLASS OUT CROSS VALIDATION
    
accuracy = np.zeros(n_cls)
#att_acc = np.zeros(n_att)

count = 0
a = 0
p = 0
r = 0
f1 = 0

y_macro_true = pd.DataFrame()
y_macro_pred = pd.DataFrame()
y_macro_true = np.array(y_macro_true)
y_macro_pred = np.array(y_macro_pred)


for i in range(0, n_cls):
    X_test_i = data[i]
    for j in range(i+1, n_cls):
        X_test_j = data[j]
        #X_test.append(X_test_j, ignore_index = True)
        count += 1
        ts_cls = [i, j]
        #print (ts_cls)
        S_test = pd.DataFrame()
        for cls in ts_cls:
           attribute_vec = aam[class_names[cls]]
           S_test = S_test.join(attribute_vec, how = 'right')
        S_test = np.array(S_test)
        
        W = get_mapping(ts_cls)
        
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        pred = evaluate(X_test_i, S_test, W, 0)
#        accuracy[i] += ai
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))
        
        
        pred = evaluate(X_test_j, S_test, W, 1)
#        accuracy[j] += aj
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.ones(pred.shape, dtype = int))
    
        a += accuracy_score(y_true, y_pred)
        p += precision_score(y_true, y_pred)
        r += recall_score(y_true, y_pred)
        f1 += f1_score(y_true, y_pred)

   
        y_true[y_true == 0] = i
        y_true[y_true == 1] = j
        y_pred[y_pred == 0] = i
        y_pred[y_pred == 1] = j
        y_macro_true = np.append(y_macro_true, y_true)
        y_macro_pred = np.append(y_macro_pred, y_pred)
        
        
        
    
        
print (count)
#accuracy = accuracy / (n_cls - 1)
#att_acc = att_acc / count
#precision = precision / count
#recall = recall / count      
    




print (accuracy)
print (accuracy.mean(axis = 0) * 100)
#print (att_acc)
print ( a/count, p/count, r/count, f1/count )

 
#%%

y_true = y_macro_true
y_pred = y_macro_pred  
print (y_true.shape, y_pred.shape, accuracy_score(y_true, y_pred))
print (classification_report(y_true, y_pred, target_names = class_names))
cnf = confusion_matrix(y_true, y_pred)
print (cnf)

plt.figure()

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
import numpy
import matplotlib
#%%

import cv2
#%%
ts_cls = [0, 3]
W = get_mapping(ts_cls)
print (W.shape)
#%%
a
#%%
a = np.ones((1, 5)) * 56
print (a)
#%%
import cv2

