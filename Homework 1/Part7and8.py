import numpy as np
from Part7support import theta
from Part7Supportlogisitc  import clf

#Make feature vector
feature_vec = np.array([6,350,80,3700,9,80,1])
feature_vec = np.reshape(feature_vec,[1,7])

#Get second order term for feature vector
X = np.reshape(np.array([1]),[1,1])
for i in range(1, 3):
    for j in range(7):
        power_col = np.power(feature_vec[:, j], i)
        power_col = np.reshape(power_col, [power_col.shape[0], 1])
        X = np.concatenate((X, power_col), axis=1)

print("Using Weights from training second order term we get the value to be ",np.dot(X,theta))

#Prediction using Logisitc regression

print("Using logisitc regression, we get the class for this feature to be",clf.predict(X))

#Part 8
#assumptions
#Cylinder = 1, disp = 80, horsepower = 1 , weight = 1800, acc = 3, model year = 4, origin = 1
feature_vec_horse = np.array([1,80,1,1800,3,40,1])
feature_vec_horse = np.reshape(feature_vec_horse,[1,7])
X = np.reshape(np.array([1]),[1,1])
for i in range(1, 3):
    for j in range(7):
        power_col = np.power(feature_vec_horse[:, j], i)
        power_col = np.reshape(power_col, [power_col.shape[0], 1])
        X = np.concatenate((X, power_col), axis=1)

print("Using Weights from training second order term we get the value of the horse cart to be ",np.dot(X,theta))
