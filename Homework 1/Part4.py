#Objective
#1. Split data set into 200 sample for training , and 192 for testing
#2. Solve using regressor function made in part 3 for polynomails from 0 to 3 using individual feature vector vs MPG
#3. Report training and testing error using OLS

import numpy as np
import pandas as pd
import Part3
from matplotlib.backends.backend_pdf import PdfPages


import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
#Import data and drop dirty data
df = pd.read_table("auto-mpg.data" , header = None,sep='\s+' , names= ["mpg" , "cylinders", "displacement", "horsepower","weight","accelaration","model_year","origin","car_name"])
df = df[df.horsepower != '?']

#Get Feature Vector and output vector
#Get Feature Vector
X = df
X = X.drop(['car_name','mpg'], axis=1)
labels = X.columns
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

print(labels)
#bump up features to respective power
def feature_bumper(feature_vec,degree):
    degree = degree+1
    X = np.ones((feature_vec.shape[0],1))
    for i in range(0, degree):
        power_col = np.power(feature_vec, i)
        X = np.concatenate((X, power_col), axis=1)
    return X
#takes in feature vector, train_labels, test_labels and prints out test and train error for each feature

def get_train_test_error(feature_vector_train, feature_vector_test, train_labels, test_labels,degree,labels):
    no_features = feature_vector_train.shape[1]
    degree = degree+1
    pp = PdfPages('Part4Plot.pdf')
    for i in range(no_features):        #iterate through features
        feature = feature_vector_train[:, i]
        feature = np.reshape(feature, [200, 1])
        feature_test = feature_vector_test[:,i]
        feature_test = np.reshape(feature_test,[192,1])
        X_newv = np.expand_dims(np.arange(min(feature_test), max(feature_test)), 1)     #initialize matrix for plotting purpose

        plt.figure(figsize=(12, 7))
        plt.scatter(feature_test,test_labels)

        for j in range(degree):         #iterate through degrees
            theta, new_feature =  Part3.linear_regressor(feature,train_labels,j)    #Train Regressor on training data
            predictions_train = Part3.hypothesis(theta,new_feature)                 #compute training predictions
            MSE_train = Part3.MSE(predictions_train, train_labels)                  #compute MSE for training data
            theta_test, new_feature_test = Part3.linear_regressor(feature_test,test_labels,j)
            predictions_test = Part3.hypothesis(theta,new_feature_test)          #compute predictions on Test data
            MSE_test = Part3.MSE(predictions_test,test_labels)                      #Compute MSE on TEST data

            X_new = feature_bumper(X_newv, j)                           #Bump up features to plot
            plot_predictions = np.dot(X_new, theta)                      #Find predictions
            plt.plot(X_newv, plot_predictions, label = j)                              #Plot predictions
            plt.xlabel(labels[i])
            plt.ylabel("MPG")
            plt.title("Polynomial lines on testing data")
            print("Training error on",labels[i],"of degree", j, "is", MSE_train)
            print("Testing error on", labels[i], "of degree", j, "is", MSE_test)
        plt.legend()
        plt.savefig(pp, format='pdf')
        plt.show()
    pp.close()

    return theta



theta = get_train_test_error(train_X,test_X,train_Y,test_Y,3,labels)






