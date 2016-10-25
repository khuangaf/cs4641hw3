'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import operator
class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.decisionTreeClassifer = []
        self.beta = []

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO  
        
        n,d = X.shape
        weight = np.ones(n) / n
        
        
        K = len(set(y)) #number of classes
        self.classes = list(set(y))
        for i in range(self.numBoostingIters):
            error = 0
            dt = DecisionTreeClassifier(max_depth = self.maxTreeDepth, splitter="random")
            dt.fit(X,y)
            predicted = dt.predict(X)
            
            for j in range(n):
                if predicted[j] != y[j]:
                    error += weight[j]
            beta = 0.5*(np.log((1.0-error)/error) + np.log(K-1.0))
            self.decisionTreeClassifer.append(dt)
            self.beta.append(beta)
            for j in range(n):
                if predicted[j] != y[j]:
                    weight[j] = weight[j] * np.exp(beta)
            weight /= np.sum(weight)
            # print weight


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        n,d = X.shape
        result= np.zeros(n)
        K = len(self.classes)
        bestClasses = np.zeros(n)
        maxCounts = np.zeros(n)
        currentCount = []

        for i in range(n):
            initCount = dict()
            for j in range(K):
                initCount[self.classes[j]] = 0
            currentCount.append(initCount)

        for i in range(self.numBoostingIters):
            predicted = self.decisionTreeClassifer[i].predict(X)
            # for k in predicted:
            #     print predicted
            # print self.beta[i]
            for j in range(n):
                for c in range(K):
                    if predicted[j] == self.classes[c]:
                        currentCount[j][predicted[j]] += self.beta[i] 
        
        for i in range(n):
            result[i] = max(currentCount[i].iteritems(), key=operator.itemgetter(1))[0]
        
        
            
        return result
