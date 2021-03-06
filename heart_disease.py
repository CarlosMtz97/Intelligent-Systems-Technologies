import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt


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
    
    #Gradient descent function
    def gradientDescent(self, hyp):
        # calculate derivatives        
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
        cost_list = [] 
        for i in range( self.epochs + 1 ) : 
            #Hypothesis
            hyp = self.hypothesis(self.X)
            
            #Cost function - Cross entropy
            cost = -1 / self.samples * np.sum(self.Y * np.log(hyp) + (1-self.Y) * np.log(1 - hyp))    

            #Gradient descent
            self.gradientDescent(hyp) 

            cost_list.append(cost)

        return cost_list
    
    def predict(self, X):
        hyp = self.hypothesis(X)        
        predict = np.where(hyp > 0.5, 1, 0)        
        return predict

#Dataset            
path = 'D:\Repos\Intelligent-Systems-Technologies\heart.csv';
df = pd.read_csv(path)
x = df[['sex','cp','exang','oldpeak','slope','ca','thal']]
y = df.target.values

X_train, X_test, Y_train, Y_test = train_test_split(x, y,test_size=0.20, random_state=5)
lr=0.001
epochs = 3000
model = Logistic_Regression( X_train, lr,epochs)
cost = model.train(X_train,Y_train)

plt.plot(np.arange(epochs + 1), cost)
plt.show()

Y_pred = model.predict( X_test ) 
    
accuracy = 0    
for i in range( np.size( Y_pred ) ) :  
    if Y_test[i] == Y_pred[i] :            
        accuracy += 1
print(f'Accuracy :  { accuracy / len(Y_pred) * 100} ')

while True:
    sex = input('Type 0 for female or 1 for male : ')
    cp = input('The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic) : ')
    exang = input('Exercise induced angina (1 = yes, 0 no) :')
    oldpeak = input('ST depression induced by exercise relative to rest (Value between 0 - 6.2) : ')
    slope = input(' Enter the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) : ')
    ca = input('Enter the numer of major vessels (0-3) : ')
    thal = input('Thalium stress test result (3 = normal; 6 = fixed defect; 7 = reversable defect) : ')

    q = pd.DataFrame(np.array([[ int(sex),int(cp),int(exang),float(oldpeak),int(slope),int(ca),int(thal)]]),columns=['sex','cp','exang','oldpeak','slope','ca','thal'])
    print(q)
    heart = model.predict(q)
    if heart[0] == 1:
        print('High risk of heart disease')
    else:
        print('Low risk of heart disease') 
