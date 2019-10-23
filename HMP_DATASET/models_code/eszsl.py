2# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:09:04 2018

@author: RAJDEEP PAL
"""

import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


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
    S = pd.DataFrame()
    Y = pd.DataFrame()
    tr_cls = [x for x in all_cls if (x not in ts_cls)]
    #print (tr_cls)
    
    
    for i, cls in enumerate(tr_cls):
        #print ('class', cls)
        df = data[cls]
        attribute_vec = aam[class_names[cls]]
        m1, d = df.shape
        
        y = np.ones((m1, n_cls-2)) * -1
        y[:, i] = 1
        y_df = pd.DataFrame(y)
        Y = Y.append(y_df, ignore_index = True)
        
        # print (m)
        S = S.join(attribute_vec, how = 'right')
        X = X.append(df, ignore_index = True)
        
    
    X = np.array(X)
    S = np.array(S)
    Y = np.array(Y)
    
    X = X.T
    (a, z) = S.shape
    (d, m) = X.shape
    
    #print (a, z, d, m)
    #print (Y.shape)
    
    
    
    i1 = np.eye(d)
    i2 = np.eye(a)
    
    Xt = X.T
    temp1 = np.matmul(X, Xt) - i1
    first_term = inv(temp1)
    #print (first_term.shape)
    
    St = S.T
    temp3 = np.matmul(S, St) + i2
    third_term = inv(temp3)
    #print (third_term.shape)
    
    
    temp2 = np.matmul(X, Y)
    second_term = np.matmul(temp2, St)
    #print (second_term.shape)
    
    V = np.matmul(first_term, second_term)
    V = np.matmul(V, third_term)
    #print (V.shape)
    return (V)
    # print ('abc', y.shape)
    
    

    
        

    





#%%

# TEST FOR A INPUT FEATURE FROM CLASS 1

def most_likely_class(x_test, ts_cls, V):
    
   from sklearn.metrics import accuracy_score
    
   x = x_test
   score = []
   for cls in ts_cls:
        # print (y_test.shape, len(y_pred))
        s = aam[class_names[cls]]
        s = np.array(s).reshape(n_att, 1)
        temp = np.matmul(x, V)
        similarity = np.matmul(temp, s)
        
        score.append(similarity)
   if score[0] > score[1]:
       return ts_cls[0]
   else:
       return ts_cls[1]
   
    
    

#%%
def evaluate(X, V, ts_cls, true_cls):
    
    
    
    (m, n) = X.shape
    y_true = np.ones(m)
    y_pred = np.zeros(m)
    for row in range(0, m):
        x_test = X[row:row+1]
        predicted_cls = most_likely_class(x_test, ts_cls, V)
#        if predicted_cls == true_cls:
#            y_pred[row] = 1
        y_pred[row] = predicted_cls
#    a = accuracy_score(y_true, y_pred)
    #p = average_precision_score(y_true, y_pred)
    #r = recall_score(y_true, y_pred)
    #print (a, p, r)
    return y_pred
        
        

#%%  LEAVE 2 CLASS OUT CROSS VALIDATION
    
a = 0
p = 0
r = 0
f1 = 0

count = 0
y_macro_true = pd.DataFrame()
y_macro_pred = pd.DataFrame()
y_macro_true = np.array(y_macro_true)
y_macro_pred = np.array(y_macro_pred)


for i in range(0, n_cls):
    X_test = data[i]
    for j in range(i+1, n_cls):
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        count += 1
        ts_cls = [i, j]
        #print (ts_cls)
        V = get_mapping(ts_cls)
        
        
        pred = evaluate(X_test, V, ts_cls, i)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))
    
        
        X_test = data[j]
        pred = evaluate(X_test, V, ts_cls, j)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.ones(pred.shape, dtype = int))
        
        y_pred[y_pred==i] = 0
        y_pred[y_pred==j] = 1
        
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
    

print ( a/count, p/count, r/count, f1/count )


#print (accuracy)
#print (accuracy.mean(axis = 0) * 100)
#print (att_acc)

 # THE END
#%%
y_true = y_macro_true
y_pred = y_macro_pred
print (y_true.shape, y_pred.shape)
print (classification_report(y_true, y_pred, target_names = class_names))
cnf = confusion_matrix(y_true, y_pred)
print (cnf)
print (accuracy_score(y_true,y_pred))

cnf = pd.DataFrame(cnf)
cnf.to_csv('cnf_eszsl.csv')
cnf.to_csv('~/ZSL/evaluation/cnf_eszsl.csv')