#######
#Objective
#1. Clean up data using LOF
#2. Clean up data using Isolation Forest
#3. Compare the two methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import keras
import functions as F
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.nan)

#LOAD DATA

#Read Data and Give labels
df = pd.read_table("yeast.data" , delim_whitespace = True , header = None)
df.columns = ["Sequence_Name", "mcg", "gvh","alm", "mit", "erl","pox","vac","nuc","class"]

#Drop Sequence Name
df = df.drop(['Sequence_Name'], axis = 1)

#Convert to Numpy array
total = df.values
total1 = np.copy(total)
#Convert class labels to logisitc labels
class_labels = ["CYT","NUC","MIT","ME3","ME2","ME1","EXC","VAC","POX","ERL"]
Y = total[:,8]
Y = Y.astype(str)
Y = np.reshape(Y,[1484,1])

#Convert labels to logisitc units
for i in range(Y.shape[0]):
    for j in range(len(class_labels)):
        if (class_labels[j] == Y[i,0]):
            Y[i,0] = j
#Get Features
total = total[:,0:8]
#Get labels
Y1 = np.copy(Y)

#Concatenate new labels with features and shuffle data
total = np.concatenate((total,Y), axis =1)
np.random.shuffle(total)

#Plot Distribution of classes
plt.figure()
ybin = np.linspace(0,600,50)
plt.hist(Y, rwidth = 0.7)
plt.yticks(ybin)
plt.grid()
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

###########################################################################
#Run LOF and clean up data

#Define LocalOutlierFactor from Sci-kit
clf = LocalOutlierFactor(n_neighbors=4, contamination=0.05)

#Extract features and labels
X = total[:,0:8]
Y = total[:,8]
Y = np.reshape(Y,[1484,1])

#Run LOF on dataset
y_pred = clf.fit_predict(X)
y_pred = np.reshape(y_pred,[1484,1])

#Convert predictions to boolean values
true_val = y_pred==1
false_val = (y_pred != 1).sum()
print("Using LOF we get %.1f outliers" %(false_val))
m = true_val.shape[0]
true_val = np.reshape(true_val,[m,1])

#Delete rows using preds
X = F.mask_delete(X,true_val)
Y = F.mask_delete(Y,true_val)

#Plot to see how data has changed
#plt.hist(Y, rwidth = 0.7)
#plt.yticks(ybin)
#plt.grid()
#plt.xlabel("Class")
#plt.ylabel("Frequency")
#plt.show()

X_embedded = TSNE(n_components=2).fit_transform(X)
a = np.reshape(X_embedded[:,0],[X_embedded.shape[0],1])
b = np.reshape(X_embedded[:,1],[X_embedded.shape[0],1])

#PLot LOF using T-Sne
plt.figure()
plt.scatter(a , b, c = Y)
plt.title("Using LOF")
plt.show()




#Using Isolation Forest to detect anomalies
IF = IsolationForest( max_samples=7, contamination = 0.05, max_features= 8, behaviour= 'new')
X1 = total1[:,0:8]
y_pred_1 = IF.fit_predict(X1)
true_val_IF = (y_pred_1 ==1)
false_val_IF = (y_pred_1 !=1).sum()
print("Using Isolation Forest we get %.1f outliers" %(false_val_IF))

Y_IF =F.mask_delete(Y1,true_val_IF)
X_IF = F.mask_delete(X1,true_val_IF)
#Uncomment to see distribution of classes
#plt.hist(Y_IF,rwidth = 0.7)
#plt.grid()
#plt.xlabel("Class")
#plt.ylabel("Frequency")
#plt.show()

#Use T-Sne to visualize
X_embedded = TSNE(n_components=2).fit_transform(X_IF)
a = np.reshape(X_embedded[:,0],[X_embedded.shape[0],1])
b = np.reshape(X_embedded[:,1],[X_embedded.shape[0],1])
#Plot using Isolation Forest using T-Sne
plt.figure()
plt.scatter(a , b, c = Y_IF)
plt.title("Using Isolation Forest")
plt.show()

#Compare the two methods
m = true_val_IF.shape[0]
true_val_IF = np.reshape(true_val_IF,[m,1])
y_pred_1 = np.reshape(y_pred_1,[1484,1])


disagree = (y_pred_1 != y_pred).sum()
print("The methods categorize %1f points different." %(disagree))




#70/30 split
#total = np.concatenate((X,Y), axis= 1)
#np.random.shuffle(total)
#m = total.shape[0]
#X = np.reshape(total[:,0:8] , [m,8])
#Y = np.reshape(total[:,8] , [m,1])

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=42)


Y_train = Y_train.astype(int)
Y_test =  Y_test.astype(int)


#Find CYT cases
mask = Y_train ==0
Xtrain_CYT = F.mask_delete(X_train,mask)
Ytrain_CYT = F.mask_delete(Y_train,mask)

#Test cases
mask_test = Y_test ==0
Xtest_CYT = F.mask_delete(X_test, mask_test)
Ytest_CYT = F.mask_delete(Y_test, mask_test)














