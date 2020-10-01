#!/usr/bin/env python
# coding: utf-8
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
    


#carico il dataset
data = pd.read_csv(r'C:\Users\Utente\anaconda3\Lib\site-packages\pandas\io\data_covnet_score-imputed_missRF_increasing_1.txt')
#print(data)

#marco
#iris = datasets.load_iris()
#target = iris.target

#assegno nome alla colonna di 0-1 e la trasformo in array
target=data['LABEL']
target=target.to_numpy()
#prendo le ultime due colonne del dataset
altri=data[data.columns[-2:]]
altri=altri.to_numpy()
#print (altri)

#marco
#sepalo=iris.data[:,:2]
#y_pred = np.ones((sepalo.shape[0], 1))

y_pred=np.ones((altri.shape[0],1))

rbf_svm = SVC(kernel='rbf')

skf = StratifiedKFold(n_splits=10)
y = target
#print(y)
#print(y_pred)
#for train_index, test_index in skf.split(sepalo, y):
for train_index, test_index in skf.split(altri, y):

    #print("TRAIN:", train_index, "TEST:", test_index)
    #print(train_index)
    X_train, X_test = altri[train_index,:], altri[test_index,:]
    #print(X_t.shape)
    #print(X_train, X_test)
    y_train, y_test = y[train_index], y[test_index]
   # print(y_test)
    clf = rbf_svm.fit(X_train, y_train)
    y_pred[test_index, 0] = clf.predict(X_test)
    #print(accuracy_score(y_test, clf.predict(X_test)))
    print(y_train,y_test)
