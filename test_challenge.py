"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
print(__doc__)
from numpy import loadtxt, ones, zeros, where
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys, traceback
from boostedDT import BoostedDT

# load the data set
filename = 'data/challengeTrainLabeled.dat'
data = loadtxt(filename, delimiter=',')

X = data[:, 0:10]
y = data[:, 10]


n,d = X.shape


# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X
ytrain = y


filename = 'data/challengeTestUnlabeled.dat'
data = loadtxt(filename, delimiter=',')

Xtest = data[:,0:10]
# ytest = y[nTrain:]

# train the decision tree
modelDT = DecisionTreeClassifier()
modelDT.fit(Xtrain,ytrain)

# train the boosted DT
modelBoostedDT = BoostedDT(numBoostingIters=1000, maxTreeDepth=3)
modelBoostedDT.fit(Xtrain,ytrain)

# output predictions on the remaining data
ypred_DT = modelDT.predict(Xtest)
ypred_BoostedDT = modelBoostedDT.predict(Xtest)
# print ypred_BoostedDT
print ','.join(str(e) for e in ypred_BoostedDT)
# print ypred_BoostedDT
# compute the training accuracy of the model
# accuracyDT = accuracy_score(ytest, ypred_DT)
# accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)

# print "Decision Tree Accuracy = "+str(accuracyDT)
# print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)