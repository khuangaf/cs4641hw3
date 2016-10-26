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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
nTrain = 0.1*n
start = 0
numofFold = 0
accuracyBoostedDT = 0.0
end = 0
while end < n:
	numofFold += 1
	end = int(start + nTrain)
	# split the data
	# print start, end, nTrain
	# print X[:start,:].shape, X[end:,:].shape
	Xtrain = np.append(X[:start,:],X[end:,:],axis=0)
	ytrain = np.append(y[:start],y[end:], axis=0)
	# print Xtrain.shape


	# filename = 'data/challengeTestUnlabeled.dat'
	# data = loadtxt(filename, delimiter=',')
	print start, end, n
	Xtest = X[start:end,:]
	ytest = y[start:end]
	# Xtest = data[:,0:10]
	# ytest = y[nTrain:]

	# train the decision tree
	modelDT = KNeighborsClassifier(n_neighbors = 1)
	modelDT.fit(Xtrain,ytrain)

	# train the boosted DT
	# modelBoostedDT = SVC(kernel = 'rbf', C= 5)
	modelBoostedDT = BoostedDT(numBoostingIters=5000, maxTreeDepth=7)
	modelBoostedDT.fit(Xtrain,ytrain)

	# output predictions on the remaining data
	ypred_DT = modelDT.predict(Xtest)
	ypred_BoostedDT = modelBoostedDT.predict(Xtest)
	# print ypred_BoostedDT
	# print ','.join(str(e) for e in ypred_BoostedDT)
	# print ypred_BoostedDT
	# compute the training accuracy of the model
	# accuracyDT = accuracy_score(ytest, ypred_DT)
	ytest = modelDT.predict(Xtest)
	accuracyBoostedDT += accuracy_score(ytest, ypred_BoostedDT)
	start += 0.1*n
# print "Decision Tree Accuracy = "+str(accuracyDT)
accuracyBoostedDT /= (numofFold)
print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)