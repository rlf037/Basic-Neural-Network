import numpy as np
from NeuralNetwork import NeuralNetwork

# === MNIST HANDWRITTEN DIGITS ===
data = np.load('mnist.npz')
X, Y = np.concatenate((data['x_train'], data['x_test']), axis=0), np.concatenate((data['y_train'], data['y_test']), axis=0)
# === NEURAL NETWORK ===
nn = NeuralNetwork(verbose=True)
nn.input(data=X, target=Y, flatten=True, problem='classification')
nn.transform('normalize')
nn.split(test_split=1/7, shuffle=True)
nn.addLayer(neurons=512, activation='relu', dropout=True)
nn.addLayer(neurons=512, activation='relu', dropout=False
nn.output(activation='softmax')
nn.compile(valid_split=1/5, optimizer='adam', scorer='accuracy', early_stoppage=True)
nn.train(batch_size=32, epochs=10)
nn.predict('argmax')
nn.evaluate()
