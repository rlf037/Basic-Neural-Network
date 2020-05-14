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
nn.addLayer(neurons=64, activation="relu", dropout=False)
nn.addLayer(neurons=128, activation="relu", dropout=True)
nn.addLayer(neurons=64, activation="relu", dropout=False)
nn.output(activation="softmax")
nn.compile(valid_split=1/10, loss='cce', optimizer="adam", scorer="accuracy", learn_rate=0.001)
nn.train(batch_size=32, epochs=10)
nn.evaluate()

nn.save('mnist')
mnist = NN.load('mnist')

import matplotlib.pyplot as plt
rand_digit = np.random.randint(0, X.shape[0])
prediction = mnist.predict(X[rand_digit])
print(f'Model: {prediction} | Actual: {Y[rand_digit]}')
plt.imshow(X[rand_digit], cmap='gray')
plt.show()