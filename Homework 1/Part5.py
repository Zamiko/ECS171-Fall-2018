import numpy as np
import pandas as pd
import Part3


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


#Function that takes in input of shape (no_of_examples, no_of_features) and returns a new weight vector with higher degree for each feature and adds a bias initially
#It also takes in the labels of shape(no_of_examples,1)
#returns new_feature_vector and theta
def linear_regressor_all(input,output,degree):
    zero_deg = degree
    degree = degree+1
    X = np.power(input[:,1],0)
    no_examples = input.shape[0]
    X = np.reshape(X,[no_examples,1])
    no_features = input.shape[1]

    #Bump up feature vector to respective power
    for i in range(1,degree):
        for j in range(no_features):
            power_col = np.power(input[:,j], i)
            power_col = np.reshape(power_col,[power_col.shape[0],1])
            X = np.concatenate((X, power_col), axis=1)
    if zero_deg == 0:           #if degree is 0, add bias column and convert everything else to ones
        X = np.ones((input.shape[0],input.shape[1]+1))

    #Use Normal Equation to solve for theta
    X_tranpose = np.transpose(X)
    theta = np.dot(X_tranpose,X)
    theta = np.linalg.pinv(theta)
    temp = np.dot(X_tranpose,output)
    theta = np.dot(theta,temp)      #Optimal Theta Value

    return  theta , X

def print_test_and_training_error(train_X,test_X,train_Y,test_Y,degree):
    degree = degree +1
    for i in range(degree):
        theta_train, X_train_new = linear_regressor_all(train_X ,train_Y, i)

        prediction_train = Part3.hypothesis(theta_train,X_train_new)
        MSE_train = Part3.MSE(prediction_train,train_Y)
        theta_test, X_test_new = linear_regressor_all(test_X,test_Y, i)
        prediction_test = Part3.hypothesis(theta_train,X_test_new)
        MSE_test = Part3.MSE(prediction_test,test_Y)
        print("Training error for degree", i, "is",MSE_train)
        print("Testing error for degree", i, "is", MSE_test)
    return X_train_new

#get input and output
train_X = total[0:200,0:7]
test_X = total[200:392,0:7]
train_Y = total[0:200,7]
train_Y = np.reshape(train_Y,[200,1])
test_Y = total[200:392,7]
test_Y = np.reshape(test_Y,[192,1])

print_test_and_training_error(train_X,test_X,train_Y,test_Y,2)