import numpy as np
import os
from NeuralNetwork import NeuralNetwork

# === MNIST HANDWRITTEN DIGITS ===
path = os.getcwd() + '/mnist.npz'
data = np.load(path)
X = np.concatenate((data['x_train'], data['x_test']), axis=0)
Y = np.concatenate((data['y_train'], data['y_test']), axis=0)

nn = NeuralNetwork()

nn.input(data=X, labels=Y)
nn.transform(Y=True, transform='categorical') #labels transform
nn.transform() #data transform
nn.split() #split data into train/test samples before training
nn.addLayer()
nn.addLayer()
nn.addLayer()
nn.output()

nn.compile()
nn.summary()
nn.train()
nn.predict()
nn.evaluate() 