# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:09:04 2018

@author: RAJDEEP PAL
"""

import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
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
feature_names = list(range(0, 25))
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

def argmaxOverMatrices(x, s, W):
    K = len(W)
    # minimum value
    best_score = -1e12
    best_idx = -1
    score = np.zeros(K)

    for i in range(0,K):
        projected_x = np.matmul(x.T, W[i])
        score[i] = np.dot(projected_x, s)
        if (score[i] > best_score):
            best_score = score[i]
            best_idx = i

    return (best_score,best_idx)



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
        
        y = np.ones((m1, 1)) * cls
        #y[:, i] = 1
        y_df = pd.DataFrame(y)
        #Y = Y.append(y_df, ignore_index = True)
        
        # print (m)
        S = S.join(attribute_vec, how = 'right')
        X = X.append(df, ignore_index = True)
        labels = labels.append(y_df, ignore_index = True)
    
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
    n_class = S.shape[1]

    # Initialization
    W = {}
    K = 10
    for i in range(0,K):
        W[i] = 1.0/np.sqrt(X.shape[1]) * np.random.rand(X.shape[1], S.shape[0])
    n_epoch = 100
    i=0
    alpha = 0.05
    
    
    
    
    # SGD
    
    for e in range(0,n_epoch):
        perm = np.random.permutation(n_train)
        for i in range(1,n_train):
            # A random image from a row
            ni = perm[i]
            best_j = -1
            # Allocate the ground truths to picked_y
            picked_y = labels[ni]
            
            # If they're same
            while(picked_y==labels[ni]):
            # Randomly generate again until those are different
                 random_index = np.random.randint(n_class)
                 picked_y = tr_cls[random_index]
        # If those are different
        # Random labeling
            picked_y = random_index
            x = X[ni, :].T.reshape(d, 1)
            col = tr_cls.index( int(labels[ni]) )
            if (picked_y == col):
                print ('corrrect', picked_y, col)
            [max_score, best_j] = argmaxOverMatrices(x, s=S[:,picked_y], W=W)
        # Grounded truth labeling
            
            
            #print (S[:, col]   ) 
            [best_score_yi, best_j_yi] = argmaxOverMatrices(x,  S[:,col],  W)
            #print (col)
            #print ( S[:, col].shape , S[:, picked_y].shape)
            if(max_score + 1 > best_score_yi):
                
                if(best_j==best_j_yi):
                    
                    W[best_j] = W[best_j] - alpha * np.matmul(x,(S[:,picked_y] - S[:,col]).reshape(1, n_att))
                else:
                
                    W[best_j] = W[best_j] - alpha * np.matmul(x , S[:,picked_y].reshape(1, n_att))
                    W[best_j_yi] = W[best_j_yi] + alpha * np.matmul(x , S[:,col].reshape(1, n_att)  )
        
    
    
    
    
    
    
    
    return W
    



    
#%%
    
def evaluate(X, W, S, ts_cls, true_cls):
    
    
    (m, n) = X.shape
#    y_true = np.ones(m)
    y_pred = np.zeros(m)
    
    
    #all_scores = []
    n_samples = m
    #n_class = len(ts_cls)

    K = len(W)
    scores = {}
    max_scores = np.zeros((K,n_samples))
    tmp_label = np.zeros((K,n_samples))
    
    

    for j in range(K):
        projected_X = np.matmul(X , W[j])
        scores[j] = np.matmul(projected_X, S)
        # Maxima along the second axis
        # Maxima between classes per an image: col
        [max_scores[j,:], tmp_label[j,:]] = [np.amax(scores[j], axis = 1),np.argmax(scores[j],axis=1)]
    # Maxima between Ws: Weight
    [best_scores, best_idx] = [np.amax(max_scores, axis=0),np.argmax(max_scores,axis=0)]

    #predict_label=np.zeros(n_samples)
    for i in range(n_samples):
        predict_label = tmp_label[best_idx[i],i]
    #    if predict_label == true_cls:
    #        y_pred[i] = 1
        y_pred[i] = predict_label
        
    #print (y_pred)        
    return y_pred


        




#print (data[0].shape)






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
    X_test_i = data[i]
    for j in range(i+1, n_cls):
        
        y_true = pd.DataFrame()
        y_pred = pd.DataFrame()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        X_test_j = data[j]
        #X_test.append(X_test_j, ignore_index = True)
        count += 1
        ts_cls = [i, j]
        #print (ts_cls)
        V = get_mapping(ts_cls)
        
        S = pd.DataFrame()
        for cls in ts_cls:
           attribute_vec = aam[class_names[cls]]
           S = S.join(attribute_vec, how = 'right')
        S = np.array(S)
        
        
        pred = evaluate(X_test_i, V, S, ts_cls, 0)
        #print (i, pred)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.zeros(pred.shape, dtype = int))
        
        pred = evaluate(X_test_j, V, S, ts_cls, 1)
        y_pred = np.append(y_pred, pred)
        y_true = np.append(y_true, np.ones(pred.shape, dtype = int))
        #print (j, pred)
        
        
        
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


ts_cls = [0, 3]
W = get_mapping(ts_cls)
K = len(W)
for i in range(0, K):
    print (W[i].shape)

#%%
    
n_train = 5
  
perm = np.random.permutation(n_train) # PEMUTATION = [0 3 1 4 2]
print (perm)
for i in range(1,n_train):
    print (i)
    
    
    
    
    
    
    
    

#%%
def complete():
    
    from scipy import sparse
    
    X = pd.DataFrame()
    S = pd.DataFrame()
    labels = pd.DataFrame()
    #print (ts_cls)
    tr_cls = all_cls
    #print (tr_cls)
    #print (tr_cls)
    
    
    for i, cls in enumerate(tr_cls):
        #print ('class', cls)
        df = data[cls]
        attribute_vec = aam[class_names[cls]]
        m1, d = df.shape
        
        y = np.ones((m1, 1)) * cls
        #y[:, i] = 1
        y_df = pd.DataFrame(y)
        #Y = Y.append(y_df, ignore_index = True)
        
        # print (m)
        S = S.join(attribute_vec, how = 'right')
        X = X.append(df, ignore_index = True)
        labels = labels.append(y_df, ignore_index = True)
    
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
    n_class = S.shape[1]

    # Initialization
    W = {}
    K = 10
    for i in range(0,K):
        W[i] = 1.0/np.sqrt(X.shape[1]) * np.random.rand(X.shape[1], S.shape[0])
    n_epoch = 15
    i=0
    alpha = 0.01
    
    
    
    
    # SGD
    
    for e in range(0,n_epoch):
        perm = np.random.permutation(n_train)
        for i in range(1,n_train):
            # A random image from a row
            ni = perm[i]
            best_j = -1
            # Allocate the ground truths to picked_y
            picked_y = labels[ni]
            
            # If they're same
            while(picked_y==labels[ni]):
            # Randomly generate again until those are different
                 random_index = np.random.randint(n_class)
                 picked_y = tr_cls[random_index]
        # If those are different
        # Random labeling
            picked_y = random_index
            x = X[ni, :].T.reshape(d, 1)
            col = tr_cls.index( int(labels[ni]) )
            if (picked_y == col):
                print ('wrong', picked_y, col)
            [max_score, best_j] = argmaxOverMatrices(x, s=S[:,picked_y], W=W)
        # Grounded truth labeling
            
            
            #print (S[:, col]   ) 
            [best_score_yi, best_j_yi] = argmaxOverMatrices(x,  S[:,col],  W)
            #print (col)
            #print ( S[:, col].shape , S[:, picked_y].shape)
            if(max_score + 1 > best_score_yi):
                
                if(best_j==best_j_yi):
                    
                    W[best_j] = W[best_j] - alpha * np.matmul(x,(S[:,picked_y] - S[:,col]).reshape(1, n_att))
                else:
                
                    W[best_j] = W[best_j] - alpha * np.matmul(x , S[:,picked_y].reshape(1, n_att))
                    W[best_j_yi] = W[best_j_yi] + alpha * np.matmul(x , S[:,col].reshape(1, n_att)  )
        
    
    
    
    
    
    
    
    
    all_scores = []
    n_samples = m
    n_class = 14

    K = len(W)
    scores = {}
    max_scores = np.zeros((K,n_samples))
    tmp_label = np.zeros((K,n_samples))
    
    

    for j in range(K):
        projected_X = np.matmul(X , W[j])
        scores[j] = np.matmul(projected_X, S)
        # Maxima along the second axis
        
        [max_scores[j,:], tmp_label[j,:]] = [np.amax(scores[j], axis = 1),np.argmax(scores[j],axis=1)+1]
    # Maxima between Ws: Weight
    [best_scores, best_idx] = [np.amax(max_scores, axis=0),np.argmax(max_scores,axis=0)]

    predict_label = np.zeros(n_samples)
    for i in range(n_samples):
        predict_label[i] = tmp_label[best_idx[i],i]
        
        
        
    
    
    # ground truth labels
    label_mat = sparse.csr_matrix((np.repeat(1,n_class),(np.squeeze(labels.reshape(1,m))-1,np.arange(n_samples))),shape=(n_class,n_samples))

    predict_mat = sparse.csr_matrix((np.repeat(1,n_class),(predict_label-1,np.arange(n_samples))),shape=(n_class,n_samples))

# predicted labels
    conf_mat = sparse.csr_matrix.dot(label_mat,np.transpose(predict_mat))
    conf_mat_diag = sparse.csr_matrix.diagonal(conf_mat)

# a kind of classes
    n_per_class = np.squeeze(np.array(np.sum(sparse.csr_matrix.transpose(label_mat),0)))


# mean class accuracy
    mean_class_accuracy = np.sum(conf_mat_diag / n_per_class) / n_class
        
            
    print (mean_class_accuracy) 
    
    
    return W
    
    
    
    
 #%%

a = complete()   
    
    
    
    
    
    
    
    
    
    
    
    
    



