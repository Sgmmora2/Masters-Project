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
from sklearn.preprocessing import StandardScaler

np.random.seed(123456789)
K=10
learn=False
auc_score = [0,0,0]

#Controls verbose mode
Do_Print = 1


#Background frequency

backgroundN = 1000

#Signal Frequency

signalN = 1000

N=signalN+backgroundN


#Define array

array = []

for i in range(backgroundN):
    array.append([0,np.random.normal(0),np.random.normal(0),np.random.normal(0),np.random.normal(0),np.random.normal(0),np.random.normal(0)])
for i in range(signalN):
    array.append([1,np.random.normal(1),np.random.normal(0),np.random.normal(1),np.random.normal(0),np.random.normal(1),np.random.normal(0)])

    
random.shuffle(array)

ROCarray = []
#create ROC array
array = pd.DataFrame(array,columns = ['label','ytest','xtest','yeval','xeval','yval','xval'])
model = sequential()
if learn == True:
    #4 layers of dense relu activated neurons with 12 on each layer
    
    model.add(dense(12, input_dim=2, activation ='relu'))
    model.add(dense(1,activation = 'sigmoid'))
    model.compile('SGD','mean_squared_error', metrics=['accuracy'])
    model.fit(array[['xtest','ytest']],array.label, epochs = 500, batch_size = 10, verbose = Do_Print,validation_data = (array[['xval','yval']],array.label))
    model.save('model')
else:
    model=load_model('model')
    
scores = model.predict(array[['xeval','yeval']])
aneuraleval,bneuraleval,cneuraleval = sklearn.metrics.roc_curve(array.label , scores)
auc_score[0] = round(sklearn.metrics.roc_auc_score(array.label,scores),3)
pyplot.plot(aneuraleval,bneuraleval)

scores = model.predict(array[['xtest','ytest']])
aneuraltest,bneuraltest,cneuraltest = sklearn.metrics.roc_curve(array.label , scores)
auc_score[1] = round(sklearn.metrics.roc_auc_score(array.label,scores),3)
pyplot.plot(aneuraltest,bneuraltest) 

aoptimal,boptimal,coptimal = sklearn.metrics.roc_curve(array.label , array.ytest)
auc_score[2] = round(sklearn.metrics.roc_auc_score(array.label,array.ytest),3) 
pyplot.plot(aoptimal,boptimal)
    
pyplot.plot([0,1])
pyplot.xlabel('FPR')
pyplot.ylabel('PR')
pyplot.legend(['neural eval auc = %s' % auc_score[0], 'neural test auc = %s' % auc_score[1], 'optimal auc = %s' % auc_score[2]])
    
pyplot.savefig('ROC for guassians seperated by 1 unit (σ=1) comparing neural network to the optimal solution.pdf', format = 'pdf')
    
