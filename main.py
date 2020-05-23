#!/Users/rlf/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork2 import NN

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data: X, Y = data['X'], data['Y']

# === HOUSE PRICES ===
# with np.load('datasets/houseprices.npz') as data: X_train, Y_train, X_test = data['X_train'], data['Y_train'], data['X_test']

# === PRE-PROCESSING ===
X = X.reshape(X.shape[0], -1)
X = NN.Normalize(X)
Y, code = NN.Encode(Y)
X_train, X_test, Y_train, Y_test = NN.Split(X, Y)
# === NEURAL NETWORK ===
model = NN(verbose=True)
model.input(input_size=X_train.shape[1])
model.hidden(neurons=100, activation='relu', dropout=0.25)
model.hidden(neurons=100, activation='relu', dropout=0.1)
model.output(output_size=len(np.unique(Y_train)), activation='softmax')
model.compile(loss='scce', learn_rate=0.01)
model.train(X_train, Y_train, batch_size=128, epochs=15, valid_split=0.1)
model.evaluate(X_test, Y_test)

# model.save('mnist')
# mnist = NN.Load('mnist')

digit = np.random.randint(0, X.shape[0])
prediction, acc = model.predict(X[digit])

print(f'Model: {code[prediction]} ({acc:.2%}) | Actual: {code[Y[digit]]}')

plt.imshow(X[digit].reshape(28, 28), cmap='bone_r')
plt.show()

# cce for one hot encoded
# scce for integers

# convert from one-hot encoding back to category integers and then decode back to original class
# with self.code dictionary that stored each class as a value to the integer key