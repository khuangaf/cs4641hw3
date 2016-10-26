'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.DNE = "does not exist"
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array

        '''
        n,d = X.shape
        # for i in range(n):
        #     print X[i,:]
        
        self.classes = list(set(y))

        K = len(self.classes)
        self.model = [] #model should be a K*d matrix with each element being a dict keeping track of post prob
        for i in range(K):
            self.model.append([])
            for j in range(d):
                self.model[i].append({})

        #determine probability of each class
        self.classProbs = np.zeros(K)   
        for i, label in enumerate(self.classes):
            count = 0
            for j in range(len(y)):
                if label == y[j]:
                    count += 1.0   
            self.classProbs[i] = count/ len(y)
        
        

        #NB        
        for i, label in enumerate(self.classes):
            # count = 0
            for j in range(n):
                if y[j] == label:
                    # count +=1
                    for k in range(d):
                        numOfDifferentValues = len(set(X[:,k]))
                        key = X[j][k]
                        if key in self.model[i][k]:
                            self.model[i][k][key] += 1
                        else:
                            self.model[i][k][key] = 1.0
        # for i in range(K):
        #     for k in range(d):
        #         print self.model[i][k]
        for i in range(K):
            for k in range(d):
                numOfDifferentValues = len(set(X[:,k]))
                totalCount = sum(self.model[i][k].values())
                #this is for reference to value that is not in the model.
                if self.useLaplaceSmoothing:
                    self.model[i][k][self.DNE] = 1.0/(numOfDifferentValues+totalCount)
                else:
                    self.model[i][k][self.DNE] = 0
                for key in self.model[i][k]:              
                    if self.useLaplaceSmoothing:
                        self.model[i][k][key] = (self.model[i][k][key] + 1.0) / (numOfDifferentValues + totalCount)
                    else:
                        self.model[i][k][key] = (self.model[i][k][key] ) / (totalCount)
    

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns: 
            an n-dimensional numpy array of the predictionss
        '''
        K = len(self.classes)
        n,d = X.shape
        result = np.zeros(n)
        for p in range(n):
            bestProb = -9999999999999
            bestClassIndex = 0
            for i in range(K):
                currentProb = self.classProbs[i]
                for k in range(d):
                    featureValue = X[p][k]
                    
                    if featureValue in self.model[i][k]:
                        currentProb += np.log(self.model[i][k][featureValue])
                    else:
                        currentProb += np.log(self.model[i][k][self.DNE])
        
                if currentProb > bestProb:
                    bestProb = currentProb
                    bestClassIndex = i
            result[p] = self.classes[bestClassIndex]
        return result

    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        K = len(self.classes)
        n,d = X.shape
        result = np.zeros([n,K])
        for p in range(n):
            probOfCurrentInstance = np.ones(K)
            for i in range(K):
                currentProb = self.classProbs[i]
                for k in range(d):
                    featureValue = X[p][k]
                    if featureValue in self.model[i][k]:
                        currentProb *= self.model[i][k][featureValue]
                    else:
                        currentProb *= self.model[i][k][self.DNE]
                probOfCurrentInstance[i] = currentProb
            #normalize
            probOfCurrentInstance /= np.sum(probOfCurrentInstance)
            result[p] = probOfCurrentInstance
        return result
    

        