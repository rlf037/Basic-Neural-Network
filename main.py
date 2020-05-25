#!/Users/rlf/anaconda3/bin/python
import numpy as np
from nnet import NN, Encoder, MinMaxScaler, Split, LoadModel
import matplotlib.pyplot as plt

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data: X, Y = data['X'], data['Y']
# === PRE-PROCESSING ===
X = X.reshape(X.shape[0], -1)
X = MinMaxScaler.transform(X)
enc = Encoder()
Y = enc.encode(Y)
X_train, X_test, Y_train, Y_test = Split.split(X, Y)
# === NEURAL NETWORK ===
model = NN(verbose=True)
model.input(input_size=X_train.shape[1])
model.hidden(neurons=512, activation='relu')
model.hidden(neurons=512, activation='relu')
model.output(output_size=10, activation='softmax')
model.compile(loss='sparse_categorical_crossentropy', learn_rate=0.01)
model.train(X_train, Y_train, batch_size=128, epochs=15, valid_split=0.1)
model.evaluate(X_test, Y_test)
model.plot()
# === SAVE & LOAD ===
# model.save('mnist')
# mnist = LoadModel('mnist')
# === PLOT PREDICTION ===
rnum = np.random.randint(0, X.shape[0])
prediction, acc = model.predict(X[rnum])
print(f'Model: {enc.decode(prediction)} ({acc:.2%}) | Actual: {enc.decode(Y[rnum])}')
img_dims = int(np.sqrt(X.shape[1]))
plt.imshow(X[rnum].reshape(img_dims, img_dims), cmap='bone_r')
plt.show()