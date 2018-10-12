import numpy as np



#MAKE A LINEAR REGRESSOR THAT CAN ACCOMODATE POLYNOMIAL BASIS FUNCTIONS ON A SINGLE VARIABLE
#Takes input array of shape(number of example,1)
#take output of shape(number of example,1)
#returns Weight Vector of shape(number of examples,1) and new feature vector to accomodate the new degrees


def linear_regressor(input,output,degree):
    degree = degree+1
    X = np.power(input,0)       #Add bias column

    #Bump up feature vector to respective power
    for i in range(0,degree):
        power_col = np.power(input,i)
        X = np.concatenate((X,power_col), axis= 1)


    #Use Normal Equation to solve for theta
    X_tranpose = np.transpose(X)
    theta = np.dot(X_tranpose,X)
    theta = np.linalg.pinv(theta)
    temp = np.dot(X_tranpose,output)
    theta = np.dot(theta,temp)      #Optimal Theta Value
    return  theta , X







#FUNCTIONS USED IN THE FUTURE -

#Takes in Input Matrix of shape (no of examples, features) and WEIGHTS of shape (no of features,1)
#returns predictions
def hypothesis(theta,X):
    output = np.dot(X, theta)
    return output

#compute mse
#Takes in predictions and outputs
#returns mse
def MSE(predictions,outputs):
    examples = predictions.shape[0]
    MSE = np.power((outputs-predictions),2)
    MSE = np.divide((np.sum(MSE)), examples)
    return MSE







