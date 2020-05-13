import numpy as np
from neuralnetwork import NN

# === MNIST HANDWRITTEN DIGITS ===
with np.load('mnist.npz') as data:
	X, Y = (
	    np.concatenate((data['x_train'], data['x_test']), axis=0),
	    np.concatenate((data['y_train'], data['y_test']), axis=0),
	)

# convert labels to strings rather than integers
Y = np.array([str(x) for x in Y])

# === NEURAL NETWORK ===
nn = NN(verbose=True)
nn.input(data=X, target=Y)
nn.transform("normalize")
nn.split(test_split=1/7, shuffle=True)
nn.addLayer(neurons=10, activation="relu", dropout=True)
nn.addLayer(neurons=15, activation="sigmoid", dropout=False)
nn.addLayer(neurons=5, activation="tanh", dropout=False)
nn.output(activation="softmax")
nn.compile(valid_split=1/10, loss='cce', optimizer="adam", scorer="accuracy", learn_rate=0.001)
nn.train(batch_size=32, epochs=10)
nn.predict()
nn.evaluate()