#Objective
#1. Split data set into 200 sample for training , and 192 for testing
#2. Solve using regressor function made in part 3 for polynomails from 0 to 3 using individual feature vector vs MPG
#3. Report training and testing error using OLS

import numpy as np
import pandas as pd
import Part3
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
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
#get input and output
train_X = total[0:200,0:7]
test_X = total[200:392,0:7]
train_Y = total[0:200,7]
train_Y = np.reshape(train_Y,[200,1])
test_Y = total[200:392,7]
test_Y = np.reshape(test_Y,[192,1])
labels = df.columns



#takes in feature vector, train_labels, test_labels and prints out test and train error for each feature

def get_train_test_error(feature_vector_train, feature_vector_test, train_labels, test_labels,degree,labels):
    no_features = feature_vector_train.shape[1]
    degree = degree+1
    for i in range(no_features):        #iterate through features
        feature = feature_vector_train[:, i]
        feature = np.reshape(feature, [200, 1])
        feature_test = feature_vector_test[:,i]
        feature_test = np.reshape(feature_test,[192,1])
        for j in range(degree):         #iterate through degrees
            theta, new_feature =  Part3.linear_regressor(feature,train_labels,j)    #Train Regressor on training data
            predictions_train = Part3.hypothesis(theta,new_feature)                 #compute training predictions
            MSE_train = Part3.MSE(predictions_train, train_labels)                  #compute MSE for training data
            theta_test, new_feature_test = Part3.linear_regressor(feature_test,test_labels,j)
            predictions_test = Part3.hypothesis(theta,new_feature_test)          #compute predictions on Test data
            MSE_test = Part3.MSE(predictions_test,test_labels)                      #Compute MSE on TEST data
            print("Training error on",labels[i],"of degree", j, "is", MSE_train)
            print("Testing error on", labels[i], "of degree", j, "is", MSE_test)



get_train_test_error(train_X,test_X,train_Y,test_Y,3,labels)






