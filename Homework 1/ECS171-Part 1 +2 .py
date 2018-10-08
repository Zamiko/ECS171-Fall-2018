#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as  pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns 



# # Objective 
# 
# ## Investigate relationship between MPG rating and its Attributes(7 attributes)
# 
# # Key Results
# 
# ## 1. Clean up data to remove unlabelled values
# 
# ## 2. Find Threshold using Normal Distribution or K-Means
# 
# ## 3. Create 2D ScatterPlot with FEATURES VS MPG
# 
# 
# 

# In[28]:


df = pd.read_table("auto-mpg.data" , header = None,sep='\s+' , names= ["mpg" , "cylinders", "displacement", "horsepower","weight","accelaration","model_year","origin","car_name"])


# In[29]:


#View datatypes and view the first view examples
df.dtypes
print(df.head())


# In[30]:


#Remove incomplete data and get X and Y values
df = df[df.horsepower != '?']
X = df
X = X.drop(['car_name','mpg'], axis=1)      #Input - drop car_name and mpg
columns = X.columns
X = X.values
X = np.reshape(X,[392,7])
Y = df.mpg          #OUTPUT
Y = Y.values        # Convert to array
Y = np.reshape(Y,[392,1])
Y = Y.astype(int)


# In[ ]:





# In[31]:


#find threshold using std distribution
std_dev = np.std(Y)   # Std Deviation
mean = np.mean(Y)     #Mean

#Get Standard Normal Distribution
distribution_val = (Y -mean)
distribution_val = np.divide(distribution_val,std_dev)
distribution_val = np.reshape(distribution_val,[392,1])

#Find Max and Min Val
max_val = np.amax(distribution_val)
min_val = np.amin(distribution_val)


#Get Range of Distribution
tot_dist = max_val - min_val


#below 33 percent of total dist goes in lower bin, 33-66 goes in mid bin, 66+ goes in upper bin 

#Get lower bin
low_threshold = min_val + ((33/100) * tot_dist)
low_val = Y[distribution_val <= low_threshold]
#print(lowlow_val.shape)
#print("max in low bin",np.amax(low_val))
#print("min in low bin",np.amin(low_val))

#get mid bin
mid_threshold = min_val + ((66/100)*tot_dist)
masks = np.logical_and(distribution_val > low_threshold, distribution_val <= mid_threshold )
mid_val = Y[masks]
#print(mid_val.shape)
#print("max in mid bin",np.amax(mid_val))
#print("min in mid bin",np.amin(mid_val))

#get high bin
high_val = Y[distribution_val > mid_threshold]
#print(high_val.shape)
#print("max in high bin",np.amax(high_val))
#print("min in high bin",np.amin(high_val))

#find thresholds by averaging distance between max and min of low,mid, and high bins
low_threshold_value = np.amax(low_val)+(np.min(mid_val)-np.amax(low_val))/2 
print("Anything below",low_threshold_value,"falls in the low MPG rating.")
mid_threshold_value = np.amax(mid_val)+(np.min(high_val)-np.amax(mid_val))/2
print("Anything above", low_threshold_value,"and below", mid_threshold_value,"falls in the mid MPG rating.")
print("Anything above", mid_threshold_value,"falls in the high MPG rating.")


# In[34]:


logistic_Y = np.copy(Y)
print(columns)


# In[35]:


#Creating 2d Scatter plots for features vs MPG
#X is a column vector with numbers
#Y is a column vector with numbers
#logistic_Y contains 0,1 or 2
#Create classes 0,1,2 for MPG Ratings
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
logistic_Y = logistic_Y.astype(int) 
X = X.astype(float)
for i in range(7):
    for j in range(7):
        a = np.reshape(X[:,i], [392,1])
        b = np.reshape(X[:,j], [392,1])
        plt.figure()
        plt.scatter(a,b,c = logistic_Y)
        plt.xlabel(columns[i])
        plt.ylabel(columns[j])
        plt.show()

        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




