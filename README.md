# Basic Neural Network

### v0.50

A simple Neural Network written in Python without the use of external libraries (except NumPy).

`neuralnetwork.py` contains the class for the Neural Network (NNet). It is an easy to use implementation in that data transformation and encoding as well as train/test split can all be done in the same class.

For classification, categorical labels must be strings and then converted to NumPy arrays (although this can be done for you automatically if possible). One-hot encoding returns and a dictionary code along the encoded labels to decode the predicted labels back into their categorical string form (check the usage).

There is the aility to save and load the model from within the class built-in as shown in the advanced usage. However it requires the dill module (pickle doesn't store lambda functions...).
Use `pip install dill` to install it.

## Usage

#### Basic
```python
from neuralnetwork import NNet

X, Y = some_dataset

model = NNet()
Y, code = NNet.encode(Y)
X = NNet.normalize(X)
X_train, X_test, Y_train, Y_test = NNet.split(X, Y)
model.input(X_train.shape)
model.hidden()
model.output(Y_train.shape)
model.compile()
model.train(X_train, Y_train)
```

#### Advanced `main.py`
```python
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
```