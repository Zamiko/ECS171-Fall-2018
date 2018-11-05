import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.nan)

#Testing_CYT
a = np.load("test_CYT.npy")     #Load Accuracy/ epoch for testing set CYT
m = a.shape[0]
a = np.reshape(a, [ m,1])
b = np.linspace(1, m, m)
b = np.reshape(b, [m,1]).astype(int)
#Plot it
test = plt.figure()
plt.plot(b, a)
plt.title("Testing Error for CYT Vs Epoch")
plt.grid()
plt.show()
test.savefig("TestError Vs Epoch.pdf") # Save FIle


a = np.load("train_CYT.npy")
m = a.shape[0]
a = np.reshape(a, [ m,1])
b = np.linspace(1, m, m)
b = np.reshape(b, [m,1]).astype(int)

test1 = plt.figure()
plt.plot(b, a)
plt.title("Training Error for CYT Vs Epoch")
plt.grid()
plt.show()

test1.savefig("TrainError Vs Epoch.pdf")