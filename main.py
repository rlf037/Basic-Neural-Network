import numpy as np
import os
from NeuralNetwork import NeuralNetwork

# === MNIST HANDWRITTEN DIGITS ===
path = os.getcwd() + '/mnist.npz'
data = np.load(path)
X = np.concatenate((data['x_train'], data['x_test']), axis=0)
Y = np.concatenate((data['y_train'], data['y_test']), axis=0)

# X = [[1, 2, 3, 2.5], 			
#  	 [2.0, 5.0, -1.0, 2.0],		
# 	 [-1.5, 2.7, 3.3, -0.8],
# 	 [-9, -8, 7, 6],
# 	 [2.0, 5.0, -1.0, 2.0],		
# 	 [-1.5, 2.7, 3.3, -0.8],
# 	 [-9, -8, 7, 6],
# 	 [2.0, 5.0, -1.0, 2.0],		
# 	 [-1.5, 2.7, 3.3, -0.8],
# 	 [-9, -8, 7, 6]]

# Y = [0, 1,2,3,4,5,6,7,8,9]

# X = np.array(X)
# Y = np.array(Y)

nn = NeuralNetwork()

nn.input(data=X, labels=Y)
nn.transform(Y=True, transform='categorical') #labels transform
nn.transform(X=True, transform='normalize') #data transform
nn.split() #split data into train/test samples before training
nn.add()
nn.add()
nn.add()
nn.output()

nn.compile()
nn.summary()
nn.train()
nn.predict()
nn.evaluate() 