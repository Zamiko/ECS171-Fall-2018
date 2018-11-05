import numpy as np
import matplotlib.pyplot as plt


Weights = np.load("WeightsCYT.npy")
X = np.load("X.npy")
Y = np.load("Y.npy")

m = Weights.shape[0]

W1 = np.reshape(Weights[:,0], [m,1])
W2 = np.reshape(Weights[:,1], [m,1])
W3 = np.reshape(Weights[:,2], [m,1])
bias = np.reshape(Weights[:,3], [m,1])
iteration = np.reshape(np.linspace(1,m,m),[m,1])


#Plot Weights
gg = plt.figure()
plt.plot(iteration, W1 , label = "W_1")
plt.plot(iteration, W2 , label = "W_2")
plt.plot(iteration, W3 , label = "W_3")
plt.plot(iteration, bias , label = "bias")
plt.legend()
plt.title("Weight changes corresponding to CYT vs Iteration ")
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.grid()
plt.show()
gg.savefig("Weights.pdf",)

#Plot bias
gg1 = plt.figure()
plt.plot(iteration, bias , label = "bias")
plt.grid()
plt.title("Bias weight vs Iteration ")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.show()
gg1.savefig("Bias.pdf")
