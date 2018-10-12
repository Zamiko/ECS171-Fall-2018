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
for i in range(logistic_Y.shape[0]):
    if logistic_Y[i]<21.3:
        logistic_Y[i] = 0

    if ((21.3<logistic_Y[i]) and (logistic_Y[i]<33.9)):
        logistic_Y[i]=1

    if(logistic_Y[i]>33.9):
        logistic_Y[i] = 2




Y = np.reshape(logistic_Y,[392,1])

#Distribute train, test set
train_X = total[0:200,0:7]
test_X = total[200:392,0:7]
train_Y = Y[0:200,0]
train_Y = np.reshape(train_Y,[200,1])
test_Y = Y[200:392,0]
test_Y = np.reshape(test_Y,[192,1])

#perform logistic_regression
clf = LogisticRegression(random_state= 0 , fit_intercept= True, solver='newton-cg', multi_class='multinomial', max_iter=1000).fit(train_X,train_Y)

#Get predictions
def get_predictions(feature_vector, predictor):
    m = feature_vector.shape[0]
    n = 1
    predictions = np.empty((m,n))
    for i in range(m):
        l = np.reshape(feature_vector[i, :], [1, 7])
        m = predictor.predict(l)
        predictions[i, 0] = m
    return predictions

#get training and testing predictions
train_predictions = get_predictions(train_X,clf)
test_predictions = get_predictions(test_X,clf)



#Define function to get precision for a label
def precision(Y_test,predictions,label):
    tp = 0
    fp =0
    no_of_examples = predictions.shape[0]

    for i in range(no_of_examples):
        if (predictions[i,0] == label and predictions[i,0] == Y_test[i,0]):
            tp = tp+1
        if (predictions[i,0] == label and predictions[i,0] != Y_test[i,0]):
            fp = fp +1
    precision = tp/(tp+fp)
    return precision

#Get Training precision/class
train_precision_0 = precision(train_Y, train_predictions, 0)
train_precision_1 = precision(train_Y, train_predictions, 1)
train_precision_2 = precision(train_Y, train_predictions, 2)
print("Training precision for class 0 is ",train_precision_0 )
print("Training precision for class 1 is ",train_precision_1 )
print("Training precision for class 2 is ",train_precision_2 )

#Get Testing precision/class
test_precision_0 =precision(test_Y,test_predictions,0)
test_precision_1 =precision(test_Y,test_predictions,1)
test_precision_2 =precision(test_Y,test_predictions,2)
print("Testing precision for class 0 is ",test_precision_0)
print("Testing precision for class 1 is ",test_precision_1)
print("Testing precision for class 2 is ",test_precision_2)