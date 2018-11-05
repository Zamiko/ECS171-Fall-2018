import numpy as np
import keras
from keras.models import  load_model

model = load_model('Model2.h5')

weights = np.array(model.get_weights())
#print(weights)
print(weights)
