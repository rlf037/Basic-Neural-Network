import numpy as np
from nnet import NeuralNetwork, LoadModel
import matplotlib.pyplot as plt

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data:
    X, Y = data['X'], data['Y']

# === PRE-PROCESSING ===
X = X.reshape(X.shape[0], -1) # flatten from 2D to 1D image data
X = NeuralNetwork.Normalize(X)

model = NeuralNetwork(verbose=True, process='classification')
Y = model.encode(Y, one_hot=False)
X_train, X_test, Y_train, Y_test = model.split(X, Y, stratify=True)
del X, Y

# === NEURAL NETWORK ===
model.input(input_size=X_train.shape[1])

model.hidden(neurons=512, activation='relu', dropout=0.1)
model.hidden(neurons=512, activation='relu')

model.output(output_size=10, activation='softmax')

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.train(X_train, Y_train, early_stopping=5)

model.evaluate(X_test, Y_test)

model.plot()

# === SAVE & LOAD ===
model.save('mnist')
model = LoadModel('mnist')

# === PLOT PREDICTION ===
rnum = np.random.randint(0, X_test.shape[0])
prediction, acc = model.predict(X_test[rnum])

print(f'Model: {model.decode(prediction)} ({acc:.2%}) | Actual: {model.decode(Y_test[rnum])}')

img_dims = int(np.sqrt(X_test.shape[1])) # convert 1D image data back to 2D for display
plt.imshow(X_test[rnum].reshape(img_dims, img_dims), cmap='bone_r')
plt.show()