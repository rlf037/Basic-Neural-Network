# Basic Neural Network

### v1.00

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
with np.load('datasets/mnist.npz') as data: X, Y = data['X'], data['Y']

# === PRE-PROCESSING ===
X = NNet.Normalize(X)
Y, code = NNet.Encode(Y)
X_train, X_test, Y_train, Y_test = NNet.Split(X, Y)
# === NEURAL NETWORK ===
model = NNet(verbose=False)
model.input(X_train.shape)
model.hidden(neurons=100, activation='ReLU')
model.output((Y_train.shape[0], 10), activation='softmax')
model.compile(loss='categorical_crossentropy', learn_rate=0.1)
model.train(X_train, Y_train, batch_size=128, epochs=15, valid_split=0.1)
model.evaluate(X_test, Y_test)

nn.save('mnist')
mnist = NN.load('mnist')

import matplotlib.pyplot as plt
rand_digit = np.random.randint(0, X.shape[0])
prediction = mnist.predict(X[rand_digit])
print(f'Model: {prediction} | Actual: {Y[rand_digit]}')
plt.imshow(X[rand_digit], cmap='gray')
plt.show()
```