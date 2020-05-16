#!/Users/rlf/anaconda3/bin/python
import numpy as np
from neuralnetwork import NNet
import matplotlib.pyplot as plt

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data:
    X, Y = data['X'], data['Y']

# === HOUSE PRICES ===
# with np.load('datasets/houseprices.npz') as data:
#     X_train, Y_train, X_test = data['X_train'], data['Y_train'], data['X_test']

X = X[:700]
Y = Y[:700]

# === PRE-PROCESSING ===
Y, code = NNet.Encode(Y)
X = NNet.Normalize(X)
X_train, X_test, Y_train, Y_test = NNet.Split(X, Y)
# === NEURAL NETWORK ===
model = NNet(verbose=False)
model.input(X_train.shape)
model.hidden(neurons=64, activation='leaky_relu', dropout=True)
model.hidden(neurons=128, activation='sigmoid', dropout=False)
model.hidden(neurons=64, activation='leaky_relu', dropout=False)
model.output(Y_train.shape, activation='softmax')
model.compile(loss='cce')
model.train(X_train, Y_train)
model.evaluate(X_test, Y_test)

# model.save('mnist')
# mnist = NNet.load('mnist')

# digit = np.random.randint(0, X.shape[0])
# prediction, acc = model.predict(X[digit])

# print(f'Model: {code[prediction]} ({acc:.2%}) | Actual: {code[np.argmax(Y[digit])]}')

# plt.imshow(X[digit], cmap='bone_r')
# plt.show()


# cce for one hot encoded
# scce for integers


# implement dropout

# convert from one-hot encoding back to category integers and then decode back to original class
# with self.code dictionary that stored each class as a value to the integer key