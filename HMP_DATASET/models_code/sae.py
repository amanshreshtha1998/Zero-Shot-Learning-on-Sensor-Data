# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:09:04 2018

@author: RAJDEEP PAL
"""

import os
import pandas as pd
import numpy as np
from scipy.linalg import solve_sylvester
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



n_cls = 14
all_cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
n_att = 300
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


#%%  ACTIVITY ATTRIBUTE MATRIX - GLOVE
    
 # F:\year 2\hpg\project\activity_attribute_matrix.csv

class_names = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk'] 

aam = pd.read_csv('F:/year 3/zsl/HMP_DATASET/models_code/activity_attribute_matrix300.csv')
print (aam.shape)



#%%
def get_mapping(ts_cls):
    
    
    
    X = pd.DataFrame()
    S = np.zeros((n_att, 1))
    #labels = pd.DataFrame()
    #print (ts_cls)
    tr_cls = [x for x in all_cls if (x not in ts_cls)]
    #print (tr_cls)
    #print (tr_cls)
    
    
    for cls in tr_cls:
        #print ('class', cls)
        df = data[cls]
        #attribute_mat = np.array([])
        #print (cls)
        attribute_vec = aam[class_names[cls]]
        #print (cls)
        m1 = df.shape[0]
        attribute_mat = np.repeat(np.array(attribute_vec).reshape(n_att, 1), m1, axis = 1)
        
        #attribute_mat = attribute_mat.reshape(n_att, m1)
        
        #y = np.ones((m1, 1)) * cls
        #y[:, i] = 1
        #y_df = pd.DataFrame(y)
        #Y = Y.append(y_df, ignore_index = True)
        
        # print (m)
        S = np.concatenate((S, attribute_mat), axis = 1)   
        X = X.append(df, ignore_index = True)
        #labels = labels.append(y_df, ignore_index = True)
    
    #print (X.shape)
    X = X.T
    X = np.array(X)
    S = S[:, 1:]
    #labels = np.array(labels)
    
    
    (k, n) = S.shape
    #print (k, n)
    (d, n) = X.shape
    #print (d, n)
    #print (labels.shape)
    labda = 1
    A = np.matmul(S, S.T)
    #print (A.shape)
    B = labda * np.matmul(X, X.T)
    #print (B.shape)
    C = (1+labda) * np.matmul(S, X.T)
    #print (C.shape)
    
    W = solve_sylvester(A, B, C)
    return (W)




#%%
def evaluate(X_test, W, S, ts_cls, true_cls):
    
    X = X_test.T
    X = np.array(X)
    
    S_pred = np.matmul(W, X)
    
    dot_matrix = cosine_similarity(S.T, S_pred.T)
    
    pred = np.argmax(dot_matrix, axis = 0)
    return pred
    
    if true_cls == 0:
        y_true = np.zeros(pred.shape, dtype = int)
        return accuracy_score(y_true, pred)
    else:
        y_true = np.ones( pred.shape, dtype = int)
        return accuracy_score(y_true, pred)
    
    






#%%  LEAVE 2 CLASS OUT CROSS VALIDATION
    

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
    mi, useless = X_test_i.shape
    for j in range(i+1, n_cls):
        
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        X_test_j = data[j]
        mj, useless = X_test_j.shape
        #X_test.append(X_test_j, ignore_index = True)
        
        count += 1
        ts_cls = [i, j]
        #print (ts_cls)
        W = get_mapping(ts_cls)
        
        S = pd.DataFrame()
        for cls in ts_cls:
           attribute_vec = aam[class_names[cls]]
           S = S.join(attribute_vec, how = 'right')
        S = np.array(S)
        
        
        pred =  evaluate(X_test_i, W, S, ts_cls, 0)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))
        
        pred = evaluate(X_test_j, W, S, ts_cls, 1)
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

#att_acc = att_acc / count
#precision = precision / count
#recall = recall / count      
    




print ( a/count, p/count, r/count, f1/count )
#print (att_acc)

 # THE END

#%%

y_true = y_macro_true
y_pred = y_macro_pred 
print (y_true.shape, y_pred.shape, accuracy_score(y_true, y_pred))
print (classification_report(y_true, y_pred, target_names = class_names))
cnf = confusion_matrix(y_true, y_pred)
print (cnf)

cnf = pd.DataFrame(cnf)
cnf.to_csv('cnf_sae.csv')
cnf.to_csv('~/ZSL/evaluation/cnf_sae.csv')
#%%

def plot_confusion_matrix(cm, title='Confusion matrix_sae', cmap=plt.cm.Oranges):
    fig, ax = plt.subplots()
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
    plt.savefig('C:/Users/RAJDEEP PAL/Desktop/abc.png')



np.set_printoptions(precision=1) 
plot_confusion_matrix(cnf)