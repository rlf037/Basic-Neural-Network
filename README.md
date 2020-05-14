# Basic Neural Network

### v0.30

A simple Neural Network written in Python without the use of external libraries (except NumPy).

`neuralnetwork.py` contains the class for the Neural Network (NN). It is an easy to use implementation in that data preprocessing/transformation, train/test split and retransformations can all be done in the same object which gives a better linear picture of the network.

All that's required is to pass X and Y to the model input and the rest of the parameters can be derived automatically based on the inputted data. Automatic parameters are input size, flattening, problem (classification or regression), transformations, train/test split, layer size, layer activations, dropout(s), output activation, output size, validation split, optimizers, scorers, early stoppages, batch sizes, epochs and retransformation. One-hot encoding is automatically applied for classification and translated back on prediction.

Warnings are also in-built if the user selects paramaters that are not optimised/recommended depending on the dataset type.

There is the aility to save and load the model from within the class built-in as shown in the advanced usage. However it requires the dill module (pickle doesn't store lambda functions...).
Use `pip install dill` to install it.

Note: For classification, the labels must be strings.

## Usage

#### Basic
```python
from neuralnetwork import NN

nn = NN()
nn.input(data=X, target=Y)
nn.split()
nn.addLayer()
nn.output()
nn.compile()
nn.train()
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