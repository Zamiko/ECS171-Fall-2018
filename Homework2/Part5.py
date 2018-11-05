import numpy as np
from tensorflow import keras
from Part1 import X
from Part1 import Y




##############################################################
#Build Keras Model to using layers and nodes

def model(layers,nodes):
    model = keras.models.Sequential()
    for i in range(layers):
        model.add(keras.layers.Dense(nodes, input_dim= 8 ,use_bias= True, activation= "sigmoid"))
    model.add(keras.layers.Dense(10,use_bias= True, activation="sigmoid"))
    sft = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer= sft,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return  model

#Run loops for layers and nodes and compute testing accuracy
def compute_test_accuracy(layers,nodes):
    for i in layers:
        for j in nodes:
            models = model(i,j)
            history = models.fit(x=X, y=Y, batch_size=1, epochs=100, validation_split=0.3, verbose= 0, shuffle= 0)
            a = np.array(history.history['val_acc'])
            final_val_acc = 1- a[99]
            print("Testing error for %f hidden layers and nodes %f is %f" %(i,j,final_val_acc))


layers = [1,2,3]
nodes = [3,6,9,12]
compute_test_accuracy(layers,nodes)









