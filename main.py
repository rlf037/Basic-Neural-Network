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

model = NeuralNetwork()

model.input(data=X, labels=Y, flatten=True)
model.transform(Y=True, transform='categorical') #labels transform
model.transform(X=True, transform='normalize') #data transform
model.split(test_split=1/7, shuffle=True, random_state=0) #split data into train/test samples before training
model.add(n_neurons=500, activation='relu')
model.add(n_neurons=500, activation='relu')
model.output(activation='softmax')

model.compile(valid_split=0.2, optimizer='adam', batch_size=128, epochs=10)
model.summary()
model.train()
model.predict()
model.evaluate() 