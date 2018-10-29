# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:02:23 2018

@author: Michael Moran Student ID: 201155224
"""

from keras.models import Sequential as sequential
from keras.layers import Dense as dense
from keras.models import Model, load_model
from keras.models import Model, save_model
import numpy as np
import random
import matplotlib.pyplot as pyplot
import sklearn.metrics
import pandas as pd
np.random.seed(123456789)
K=10
learn=False
auc_score_array = np.empty(K)

#Controls verbose mode
Do_Print = 0

for distance in range(K):

    
    #Background frequency
    
    backgroundN = 150
    
    #Signal Frequency
    
    signalN = 150
    
    N=signalN+backgroundN
    
    #Define Array
    
    array = []
    evalarray=[]
    
    for i in range(backgroundN):
        array.append([0,np.random.normal()])
    for i in range(signalN):
        array.append([1,np.random.normal(distance)])
    random.shuffle(array)
    
    for i in range(backgroundN):
        evalarray.append([0,np.random.normal()])
    for i in range(signalN):
        evalarray.append([1,np.random.normal(distance)])
    random.shuffle(evalarray)
    
    ROCarray = []
    #create ROC array
    
    #for i in np.arange(-5,5,0.05):
    #    falsepositive = 0
    #    positive = 0
    #    for j in range(N):
    #        if i<array[j][1]:
    #            if array[j][0] == 'background':
    #                falsepositive+=1
    #            if array[j][0] == 'signal':
    #                positive+=1
    #    ROCarray.append([falsepositive/backgroundN,positive/signalN])
    #
    #ROCarray = np.array(ROCarray)    
    #pyplot.scatter(ROCarray[:,0],ROCarray[:,1])
    #pyplot.plot([0,1])
    #pyplot.xlabel('false positive')
    #pyplot.ylabel('positive')
    #pyplot.savefig('ROCgaussian')
    
    array = pd.DataFrame(array,columns = ['label','value'])
    evalarray = pd.DataFrame(evalarray,columns = ['label','value'])
    model = sequential()
    if learn == True:
        #4 layers of dense sigmoid activated neurons with 12 on each layer
        
        model.add(dense(12, input_dim=1, activation ='sigmoid'))
        model.add(dense(12, activation='sigmoid'))
        model.add(dense(12, activation='sigmoid'))
        model.add(dense(12, activation='sigmoid'))
        model.add(dense(1,activation = 'sigmoid'))
        model.compile('SGD','mean_squared_error', metrics=['accuracy'])
        model.fit(array.value,array.label, epochs = 20, batch_size = 1, verbose = Do_Print)
        model.save('model')
    else:
        model=load_model('model')
    scores = model.predict(evalarray.value)
    scores.resize(300,)
    a,b,c = sklearn.metrics.roc_curve(evalarray.label , scores)
    auc_score_array[distance] = round(sklearn.metrics.roc_auc_score(evalarray.label,scores),3)
    pyplot.plot(a,b)
        
    
pyplot.plot([0,1])
pyplot.xlabel('FPR')
pyplot.ylabel('PR')
    
    
pyplot.legend(['Distance %s auc_score %s' % (i,auc_score_array[i]) for i in range(K)])
pyplot.savefig('ROC for guassians (σ=1) of varying seperation neural net version')
    
