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
import itertools
import matplotlib.pyplot as plt

n_cls = 6
all_cls = [0, 1, 2, 3, 4, 5]
n_att = 300
seed = 0


#%% LOAD DATASET

directory = 'F:/year 3/zsl/HAR_DATASET/extracted_features/final_extracted_features_32'
arr = os.listdir(directory)

#feature_names = ['maxX', 'minX', 'avgX', 'stdX', 'slopeX', 'zcrX',     'maxY','minY','avgY','stdY', 'slopeY', 'zcrY',          'maxZ', 'minZ', 'avgZ', 'stdZ', 'slopeZ', 'zcrZ',      'maxACC', 'minACC', 'avgACC', 'stdACC',     'XYcorr', 'YZcorr','ZXcorr',    'energy']
#feature_names = ['0', '1', '2', '3', '4', '5',     '6','7','8','9', '10', '11',          '13', '14', '15', '16', '17', '18',      '19', '20', '21', '22',     '23', '24','25',    '26', '27', '28', '29', '30', '31']
#frature_names = list(range(0, 32))
path = directory+'/'+arr[0]
data = []


for file_name in arr:
    path = directory + '/' + file_name
    #df  = pd.read_csv(path, index_col = False, names = feature_names)
    df  = pd.read_csv(path)
    data.append(df)
    print (df.shape)

print (data[0])

#%%  ACTIVITY ATTRIBUTE MATRIX - GLOVE
    
 # F:\year 2\hpg\project\activity_attribute_matrix.csv

class_names = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying']

aam = pd.read_csv('F:/year 3/zsl/HAR_DATASET/activity_attribute_matrix300.csv')
print (aam.shape)

#%%
def get_clf(ts_cls):
    from sklearn.svm import SVC, OneClassSVM
    
    X = pd.DataFrame()
    y = pd.DataFrame()
    tr_cls = [x for x in all_cls if (x not in ts_cls)]
    
    
    for cls in tr_cls:
        #print ('class', cls)
        df = data[cls]
        m, n = df.shape
        # print (m)
        
        
        tgt_df = pd.DataFrame(np.ones( (m,1), dtype = int) * cls)
            #print (1)
        
            #print (0)
            
        X = X.append(df, ignore_index = True)
        y = y.append(tgt_df, ignore_index = True)
        
    # print ('abc', y.shape)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    clf = SVC(probability = True).fit(X, y)
    labels = clf.classes_
    
    return (clf, labels)



#%%
    
def most_likely_class(best_T, S, labels, prob):
    
    T = 3
    prob = prob.T
    pred_att = np.zeros(n_att).reshape(n_att, 1)
    sum_p = 0;
        
    
    for i in range(T):
        t = best_T[0][i]
        #print ('acv', labels)
        tr_cls_t = labels[t][0]
        #print ('abc', tr_cls_t)
        p = prob[t][0]
        s = np.array (aam[class_names[tr_cls_t]]).reshape(n_att, 1)
        sum_p += p
        pred_att = np.add(pred_att, p*s)
    pred_att = pred_att/sum_p  
    dot_matrix = cosine_similarity(S.T, pred_att.T)
    pred_cls = np.argmax(dot_matrix, axis = 0)
    return pred_cls[0]
    
        

#%%
    
def evaluate(X_test, S, ts_cls, true_cls):
    
    (m, d) = X_test.shape
    y_pred = np.zeros((m, 1), dtype = int)
    (clf, labels) = get_clf(ts_cls)
    labels = np.array(labels).reshape(n_cls-2, 1)
    #print (labels)
    
    for row in range(m):
        x = X_test[row:row+1]
        prob =  np.array ( clf.predict_proba(x)).reshape(1, n_cls-2) 
        best_T = np.argsort(-1 *prob, axis=1)
        y_pred[row] = most_likely_class(best_T, S, labels, prob)
    
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
    X_test_i = data[i]
    for j in range(i+1, n_cls):
        X_test_j = data[j]
        #X_test.append(X_test_j, ignore_index = True)
        count += 1
        ts_cls = [i, j]
        #print (ts_cls)
        
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        
        S = pd.DataFrame()
        for cls in ts_cls:
           attribute_vec = aam[class_names[cls]]
           S = S.join(attribute_vec, how = 'right')
        S = np.array(S)
        
        
        pred = evaluate(X_test_i, S, ts_cls, 0)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))
        
        pred = evaluate(X_test_j, S, ts_cls, 1)
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
    

print ( a/count, p/count, r/count, f1/count )


#print (accuracy)
#print (accuracy.mean(axis = 0) * 100)
#print (att_acc)

 # THE END
print (X_test_i.shape)
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
ts_cls = [0, 3]
(a, b) = get_clf(ts_cls)
print (a, b)




#%%

ans = df.nlargest(4, ['a', 'b', 'c'])
print (ans)

#%%

probs = np.array( [[5, 4, 3, 6, 7, 10, 88],
                 [1, 2, 3, 4, 5, 6, 7]]).reshape(2, 7)
print (probs)
n = 3
best_n = np.argsort(-1*probs, axis=1)
print (best_n)
print ( best_n[1][4] )

a = range(0, 32)
print (list(a))

