import numpy as np
from NeuralNetwork import NeuralNetwork

X_train = [[1, 2, 3, 2.5], 			
 	 [2.0, 5.0, -1.0, 2.0],		
	 [-1.5, 2.7, 3.3, -0.8],
	 [-9, -8, 7, 6]]

Y_train = [3, 1, 2, 0]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

model = NeuralNetwork()

model.add(layer='input', data=X_train, labels=Y_train)
model.add(layer='transform', transform='categorical') #labels transform
model.add(layer='transform', transform='normalize') #data transform
model.add(layer='hidden', n_neurons=128)
model.add(layer='hidden', n_neurons=128)
model.add(layer='output')

model.summary()
model.compile()
model.fit()
model.predict()
model.evaluate() 