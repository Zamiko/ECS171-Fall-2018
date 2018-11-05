#Objective
#1. Build Model
#2. Train Data
#3. Plot weight and Bias for class CYT from last layer
#4. Training and testing error for Class CYT and Full Model

import matplotlib.pyplot as plt
from tensorflow import keras
from Part1 import X_train, X_test, Y_train, Y_test, Xtrain_CYT, Xtest_CYT, Ytrain_CYT, Ytest_CYT
import numpy as np





##############################################################
#Build Keras Model

def model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(3, input_dim= 8 ,use_bias= True, activation= "sigmoid", name= 'Hidden1'))
    model.add(keras.layers.Dense(3,use_bias= True, activation="sigmoid", name='Hidden2'))
    model.add(keras.layers.Dense(10,use_bias= True, activation="sigmoid", name='output'))
    sft = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer= sft,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return  model

#Initialize Model
models = model()


#Define function to save weights for class CYT

def save_CYT_weights(mod):
    weight = mod.get_weights()  #Get weights
    a = np.array(weight)
    W3 = a[4]            # Get weights for W3
    W3 = np.transpose(W3)  #Transpose W3
    W3 = W3[0, :]           #Extract weights for class CYT
    bias = a[5]             #Extract bias corresponding to last layer
    bias = np.array([bias[0]])  #Extract bias corresponding to CYT
    d = np.reshape(np.append(W3, bias, axis=0),[1,4]) #Return Weights + Bias
    return d

#Define function to find training and testing error for class CYT
def train_error(mod, X_train, X_test, Y_train, Y_test):
    #Get prediction values
    prediction_train = np.argmax(mod.predict(X_train), axis= 1)     #Find predictions for training
    prediction_test = np.argmax(mod.predict(X_test), axis=1 )       #Find predictions for testing
    #Get total size
    m = prediction_train.shape[0]                       # Get number of examples of training
    m2 = prediction_test.shape[0]                       # Get number of examples of testing
    #Reshape
    prediction_train = np.reshape(prediction_train, [m,1])
    prediction_test = np.reshape(prediction_test, [m2,1])
    acc = (prediction_train == Y_train).sum()   #FInd all true values
    error = 1 - (acc/m)                           #Find train error
    acc_test = (prediction_test == Y_test).sum()    # Find test error
    error_test = 1 - (acc_test/m2)
    return error

#Find testing error for class CYT
#Returns testing error
def test_error(mod, X_train, X_test, Y_train, Y_test):
    prediction_train = np.argmax(mod.predict(X_train), axis= 1)
    prediction_test = np.argmax(mod.predict(X_test), axis=1 )
    m = prediction_train.shape[0]
    m2 = prediction_test.shape[0]
    prediction_train = np.reshape(prediction_train, [m,1])
    prediction_test = np.reshape(prediction_test, [m2,1])
    acc = (prediction_train == Y_train).sum()

    acc_test = (prediction_test == Y_test).sum()
    error_test = 1- (acc_test/m2)
    return error_test



#Convert to One hot encoding
Y_train_1 = keras.utils.to_categorical(Y_train, num_classes= 10)
Y_test_1 = keras.utils.to_categorical(Y_test, num_classes= 10)


#Callbacks
#Save Model Weight corresponding to activation CYT
Weight_CYT = []
tracker_w = keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: Weight_CYT.append(save_CYT_weights(models)) )
#Save Training error for CYT
train_CYT_error = []
tracker_train = keras.callbacks.LambdaCallback(on_epoch_begin= lambda epoch, logs: train_CYT_error.append(train_error(models,Xtrain_CYT,Xtest_CYT,Ytrain_CYT,Ytest_CYT)))
#Save testing error for CYT
test_CYT_error = []
tracker_test = keras.callbacks.LambdaCallback(on_epoch_begin= lambda epoch, logs: test_CYT_error.append(test_error(models,Xtrain_CYT,Xtest_CYT,Ytrain_CYT,Ytest_CYT)))

#Train Model
history = models.fit(x = X_train, y = Y_train_1, batch_size= 1, epochs = 300, validation_split= 0, validation_data = (X_test,Y_test_1)  ,callbacks = [tracker_w, tracker_train, tracker_test], shuffle= False , verbose = 2)

#Save Model
models.save('Model2.h5')


#Convert list to array and reshape
m = np.array(Weight_CYT).shape[0]
weight_P2 = np.reshape(Weight_CYT , [m,4])

#Save arrays
np.save("WeightsCYT", weight_P2 )
np.save("train_CYT", train_CYT_error)
np.save("test_CYT", test_CYT_error)



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for error
error = np.array(history.history['acc'])
m = error.shape[0]
error_train = np.reshape(error, [m,1])
error_train = 1 - error_train
plt.plot(error_train)
error_test = np.array(history.history['val_acc'])
m = error_test.shape[0]
error_test = np.reshape(error_test, [m,1])
error_test = 1 - error_test
plt.plot(error_test)

plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['error_train', 'error_test'], loc='upper left')
plt.show()





