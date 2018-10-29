# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:02:23 2018

@author: Michael Moran Student ID: 201155224
"""
import numpy as np
import random
import matplotlib.pyplot as pyplot
import sklearn.metrics
import pandas as pd
N=10
auc_score_array = np.empty(N)
for distance in range(N):

    
    #Background frequency
    
    backgroundN = 150
    
    #Signal Frequency
    
    signalN = 150
    
    N=signalN+backgroundN
    
    #Define Array
    
    array = []
    
    for i in range(backgroundN):
        array.append([0,np.random.normal()])
    for i in range(signalN):
        array.append([1,np.random.normal(distance)])
    random.shuffle(array)
    
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
    a,b,c = sklearn.metrics.roc_curve(array.label, array.value)
    np.append(auc_score_array,[(sklearn.metrics.roc_auc_score(array.label,array.value))])
    pyplot.plot(a,b)
    
pyplot.plot([0,1])
pyplot.xlabel('FPR')
pyplot.ylabel('PR')
    
    
pyplot.legend(['Distance %s auc_score %s' % (i,auc_score_array[i]) for i in range(10)])
pyplot.savefig('ROC for guassians (σ=1) of varying seperation')
    
