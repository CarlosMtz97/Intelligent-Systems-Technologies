from itertools import accumulate
from os import path
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 


#Logistic Regression
class Logistic_Regression() :
    def __init__(self,X, lr, epochs):
        self.lr = lr        
        self.epochs = epochs
        self.samples, self.features = X.shape              
        self.weights = np.zeros(self.features)        
        self.bias = 0   

    #Sigmoid function
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    #Hypothesis function
    def hypothesis(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias) 
    
    #Gradient descent
    def gradientDescent(self, hyp):
        # calculate gradients        
        dw = 1 / self.samples * np.dot(self.X.T,hyp-self.Y.T)
        db = 1 / self.samples * np.sum(hyp-self.Y.T) 
          
        #Update weight values  
        self.weights -= self.lr*dw
        self.bias -= self.lr*db
          
        return self
    
    #Training function
    def train(self, X, Y):     
        self.X = X        
        self.Y = Y
          
        #Gradient descent function          
        for i in range( self.epochs + 1 ) : 
            hyp = self.hypothesis(self.X)
            
            #Cost function - Cross entropy
            cost = -1 / self.samples * np.sum(self.Y * np.log(hyp) + (1-self.Y) * np.log(1 - hyp))    

            self.gradientDescent(hyp) 
            
            if i % 1000 == 0:
                print(f'Cost after {i} epochs : {cost}')           
        return self
    
    def predict(self, X):
        hyp = self.hypothesis(X)        
        predict = np.where(hyp > 0.5, 1, 0)        
        return predict

            
path = 'D:\Repos\Intelligent-Systems-Technologies\heart.csv';
df = pd.read_csv(path)
x = df[['sex','cp','fbs','exang','oldpeak','slope','ca','thal']]
y = df.target.values
X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=0.20, random_state=42)
lr=0.1
epochs = 3000
logreg = Logistic_Regression( X_train, lr,epochs)
logreg.train(X_train,Y_train)

Y_pred = logreg.predict( X_test ) 
    
accuracy = 0    
for i in range( np.size( Y_pred ) ) :  
    if Y_test[i] == Y_pred[i] :            
        accuracy += 1
print(f'Accuracy :  { accuracy / len(Y_pred) * 100} ')

    