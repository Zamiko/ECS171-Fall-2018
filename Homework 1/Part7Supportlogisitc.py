#Goals
#Use logistic regression for classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

#Import data and drop dirty data
df = pd.read_table("auto-mpg.data" , header = None,sep='\s+' , names= ["mpg" , "cylinders", "displacement", "horsepower","weight","accelaration","model_year","origin","car_name"])
df = df[df.horsepower != '?']

#Get Feature Vector and output vector
#Get Feature Vector
X = df
X = X.drop(['car_name','mpg'], axis=1)
X = X.values

#Get output Vector
Y = df
Y = Y.mpg
Y = Y.values
Y = np.reshape(Y, [392,1])
total = np.concatenate((X,Y), axis=1)

#shuffle matrix
np.random.shuffle(total)
total = total.astype(float)
Y = total[:,7]
logistic_Y = np.copy(Y)

#create logisitic units
zero_term = 0
one_term = 0
two_term = 0
for i in range(logistic_Y.shape[0]):
    if logistic_Y[i]<21.3:
        logistic_Y[i] = 0
        zero_term += 1
    if ((21.3<logistic_Y[i]) and (logistic_Y[i]<33.9)):
        logistic_Y[i]=1
        one_term += 1
    if(logistic_Y[i]>33.9):
        logistic_Y[i] = 2
        two_term +=1



Y = np.reshape(logistic_Y,[392,1])

#Distribute train, test set
train_X = total[0:200,0:7]
test_X = total[200:392,0:7]
train_Y = Y[0:200,0]
train_Y = np.reshape(train_Y,[200,1])
test_Y = Y[200:392,0]
test_Y = np.reshape(test_Y,[192,1])
X = np.ones((200,1))

#Second order feature_vector
for i in range(1, 3):
    for j in range(7):
        power_col = np.power(train_X[:, j], i)
        power_col = np.reshape(power_col, [power_col.shape[0], 1])
        X = np.concatenate((X, power_col), axis=1)



#perform logistic_regression
clf = LogisticRegression(random_state= 0 , fit_intercept= False, solver='newton-cg', multi_class='multinomial', max_iter=1000).fit(X,train_Y)


