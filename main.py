import numpy as np
from neuralnetwork import NN

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist2.npz') as data:
    X, Y = data['X'], data['Y']

# convert labels to strings rather than integers
# Y = np.array([str(x) for x in Y])

# === NEURAL NETWORK ===
nn = NN(verbose=True)
nn.input(data=X, target=Y)
nn.transform()
nn.split()
nn.addLayer()
nn.addLayer()
nn.addLayer()
nn.output()
nn.compile()
nn.train()
nn.evaluate()

# nn.save('mnist')
# mnist = NN.load('mnist')

# import matplotlib.pyplot as plt
# digit = np.random.randint(0, X.shape[0])
# prediction, acc = mnist.predict(X[digit])
# print(f'Model: {prediction} ({acc:.2%}) | Actual: {Y[digit]}')
# plt.imshow(X[digit], cmap='bone_r')
# plt.show()

# cce for one hot encoded
# scce for integers