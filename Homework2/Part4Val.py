import numpy as np
np.set_printoptions(threshold=np.nan)

def sigmoid(x, derivative=False):
  val = np.divide(1, 1+np.exp(-x))
  return val

def sigmoid_derivative(x):
  val = x* (1-x)
  return val

X = np.load("X_hand.npy").astype(float)
Y = np.load("Y_hand.npy")
weights = np.load("Weights_Hand.npy")



a = np.array(weights[0][0])


#Layer 1
Layer_1 = sigmoid(np.reshape(np.dot(np.transpose(a),X[0]),[3,1]))



a = np.array(weights[0][2])



#Layer 2
Layer_2 = sigmoid(np.reshape(np.dot(np.transpose(a),Layer_1), [ 3,1]))
a = np.array(weights[0][4])


#Layer 3

Layer_3 = sigmoid(np.reshape(np.dot(np.transpose(a),Layer_2), [ 10,1]))
#print(Layer_3)     #Activation
#Delta3
delta3 = (Layer_3 -  np.reshape(Y[0], [ 10,1]))

#Dl/ Dw3
dldw3 = np.dot(Layer_2, np.transpose(delta3))
#Update W3
W3_new = np.array(weights[0][4]) - (0.01*dldw3)
#Compare new values of W3 from manual math and Computation
print("Weight 3 from updating manually")
print(W3_new)
print("Weight 3 from Keras")
print(np.array(weights[1][4]))
print("Bias 3 from updating manually ")
B3_new = np.reshape(np.array(weights[0][5]) , [10,1]) - (0.01*delta3)
print(B3_new)
print("Bias 3 form Keras ")
print(np.array(weights[1][5]))
#################################################################
W3 = np.array(weights[0][4])
delta2 = np.multiply(np.dot(W3,delta3) , sigmoid_derivative(Layer_2))
print("A1")
print(Layer_1)
dldw2 = np.dot(delta2, np.transpose(Layer_1))
print("Value from updating man")
W2_new = np.array(weights[0][2]) - (0.01*dldw2)
print(W2_new)
print("Value from updating layer 2")
print(np.array(weights[0][2]))