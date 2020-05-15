#!/Users/rlf/anaconda3/bin/python

import numpy as np
from neuralnetwork import NNet

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data:
    X, Y = data['X'], data['Y']

# X = X[:7000]
# Y = Y[:7000]

# === NEURAL NETWORK ===
model = NNet(verbose=True)
Y, code = NNet.encode(Y)
X = NNet.normalize(X)
X_train, X_test, Y_train, Y_test = NNet.split(X, Y)
model.input(X_train.shape)
model.hidden(neurons=10)
model.hidden(neurons=100, dropout=True)
model.hidden(neurons=30)
model.output(Y_train.shape, activation='softmax')
model.compile(batch_size=100)
model.train(X_train, Y_train)
model.evaluate(X_test, Y_test)

# model.save('mnist')
# mnist = NNet.load('mnist')

# import matplotlib.pyplot as plt
# digit = np.random.randint(0, X.shape[0])
# prediction, acc = mnist.predict(X[digit])
# print(f'Model: {prediction} ({acc:.2%}) | Actual: {code[np.argmax(Y[digit])]}')
# plt.imshow(X[digit], cmap='bone_r')
# plt.show()

# cce for one hot encoded
# scce for integers


#implement dropout

        #convert from one-hot encoding back to category integers and then decode back to original class
        # with self.code dictionary that stored each class as a value to the integer key