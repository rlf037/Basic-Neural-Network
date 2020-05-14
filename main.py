import numpy as np
from neuralnetwork import NN

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist2.npz') as data:
    X, Y = data['X'], data['Y']

# convert labels to strings rather than integers
# Y = np.array([str(x) for x in Y])
# X = X[:1000]
# Y = Y[:1000]
# === NEURAL NETWORK ===
nn = NN(verbose=True)
Y = nn.encode(Y)
X = NN.normalize(X)
X_train, X_test, Y_train, Y_test = NN.split(X, Y)
nn.input(X_train.shape)
nn.addLayer(neurons=10)
nn.output(Y_train.shape, activation='softmax')
nn.compile(batch_size=100)
nn.train(X_train, Y_train)
nn.evaluate(X_test, Y_test)

# nn.save('mnist')
# mnist = NN.load('mnist')

# import matplotlib.pyplot as plt
# digit = np.random.randint(0, X.shape[0])
# prediction, acc, code = mnist.predict(X[digit])
# print(f'Model: {prediction} ({acc:.2%}) | Actual: {code[np.argmax(Y[digit])]}')
# plt.imshow(X[digit], cmap='bone_r')
# plt.show()

# cce for one hot encoded
# scce for integers