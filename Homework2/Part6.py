import numpy as np
from tensorflow import keras
from Part1 import X
from Part1 import Y



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

models = model(1,12)

models.fit(x=X, y=Y, batch_size=1, epochs=100, validation_split=0.3, verbose= 1, shuffle= 0)

data = np.array([0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39])
data = np.reshape(data, [1,8])
a = models.predict_classes(data)
b = models.predict(data)
print("The class is %.1f" %a )
print(b)
print(1-b)

