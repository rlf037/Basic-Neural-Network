import numpy as np
from NeuralNetwork import NeuralNetwork

X = [[1, 2, 3, 2.5], 			
 	 [2.0, 5.0, -1.0, 2.0],		
	 [-1.5, 2.7, 3.3, -0.8],
	 [-9, -8, 7, 6]]

Y = [1, 0, 1, 0]

X = np.array(X)
Y = np.array(Y)

model = NeuralNetwork()

model.add(layer='input', data=X)
model.add(layer='transform', transform='standardize')
model.add(layer='hidden', n_neurons=100)
model.add(layer='hidden', n_neurons=50)
model.add(layer='hidden', n_neurons=100)
model.add(layer='output', data=Y)

model.summary()
model.compile()
model.fit()
model.predict()
model.evaluate() 