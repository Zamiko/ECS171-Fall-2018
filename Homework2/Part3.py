import numpy as np
import keras
import pandas as pd
from keras.utils import  to_categorical

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

#Get all features
X = total[:,0:8]
Y = total[:,8]
Y = np.reshape(Y,[1484,1])
Y = to_categorical(Y)

#Define model
def model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(3, input_dim= 8 ,use_bias= True, activation= "sigmoid"))
    model.add(keras.layers.Dense(3, use_bias= True, activation="sigmoid"))
    model.add(keras.layers.Dense(10, use_bias= True, activation="sigmoid"))
    sft = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer= sft,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return  model

models = model()
history = models.fit(x = X, y = Y, batch_size= 1, epochs = 100, validation_split= 0, verbose=2)
#Save model
models.save('Model3.h5')

