#!/Users/rlf/anaconda3/bin/python
import numpy as np
from neuralnetwork2 import NN, LoadModel

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data: X, Y = data['X'], data['Y']

# === HOUSE PRICES ===
# with np.load('datasets/houseprices.npz') as data: X_train, Y_train, X_test = data['X_train'], data['Y_train'], data['X_test']

# X = X[:1000]
# Y = Y[:1000]

# === PRE-PROCESSING ===
X = X.reshape(X.shape[0], -1)
X = NN.Normalize(X)
Y, code = NN.Encode(Y)
X_train, X_test, Y_train, Y_test = NN.Split(X, Y)
# === NEURAL NETWORK ===
model = NN(verbose=True)
model.input(input_size=X_train.shape[1])
model.hidden(neurons=50, activation='relu')
model.hidden(neurons=50, activation='relu')
model.output(output_size=10, activation='softmax')
model.compile(loss='scce', learn_rate=0.01)
model.train(X_train, Y_train, batch_size=128, epochs=15, valid_split=0.1)
model.evaluate(X_test, Y_test)

# model.save('mnist')
# mnist = LoadModel('mnist')

# digit = np.random.randint(0, X.shape[0])
# prediction, acc = mnist.predict(X[digit])
# print(f'\nModel: {code[prediction]} ({acc:.2%}) | Actual: {code[Y[digit]]}')
# import matplotlib.pyplot as plt
# plt.imshow(X[digit].reshape(28, 28), cmap='bone_r')
# plt.show()