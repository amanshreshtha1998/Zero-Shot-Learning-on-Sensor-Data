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

n_cls = 6
all_cls = [0, 1, 2, 3, 4, 5]
n_att = 9
seed = 0


#%% LOAD DATASET

directory = 'F:/year 3/zsl/HAR_DATASET/extracted_features/final_extracted_features_32'
arr = os.listdir(directory)

#feature_names = ['0', '1', '2', '3', '4', '5',     '6','7','8','9', '10', '11',          '13', '14', '15', '16', '17', '18',      '19', '20', '21', '22',     '23', '24','25',    '26', '27', '28', '29', '30', '31']
feature_names = list(range(0, 25))
path = directory+'/'+arr[0]
data = []


for file_name in arr:
    path = directory + '/' + file_name
    df  = pd.read_csv(path, index_col = False, names = feature_names)
    #df  = pd.read_csv(path)
    data.append(df)
    print (df.shape)

print (data[0])


#%% LOAD ACTIVITY ATTRIBUTE MATRIX
    
 # F:\year 2\hpg\project\activity_attribute_matrix.csv
att_names = ['sittting', 'standing', 'walking', 'posture_upright','arm_pendulum_swing', 'translation_motion', 'cyclic_motion', 'potential_energy_increase', 'potential_energy_decrease']
aam = pd.read_csv('F:/year 3/zsl/HAR_DATASET/code/matrix1.csv', names = att_names)
print (aam)
F:\year 3\zsl\HAR_DATASET\code

class_names = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying']



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
    #print (X.shape, y.shape)
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

y_macro_true = pd.DataFrame()
y_macro_pred = pd.DataFrame()
y_macro_true = np.array(y_macro_true)
y_macro_pred = np.array(y_macro_pred)

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
        y_true = np.append( y_true, np.zeros(pred.shape, dtype=int))
        
        
                                                                                                                                                                                                                                                  
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




#%%
y_true = y_macro_true
y_pred = y_macro_pred 
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















