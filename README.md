# Basic Neural Network

### v1.10

A simple Neural Network written in Python usually only NumPy.

#### TODO:

- Add in a stratify feature in split()

Neural Network:  
    - Activations: `relu` `leaky_relu` `tanh` `sigmoid` `softmax`  
    - Loss Functions: `spare_categorical_crossentropy` `categorical_crossentropy` `mse` `mae`  
    - Optimizers: `adam` `rmsprop` `adadelta` `sgd`  

Functions:  
    - `plot` Plots the model training loss and accuracy.  
    - `save` Saves a model's weights & biases.  
    - `evaluate` Evaluates a model's accuracy.
    - `predict` Returns an encoded prediction.

PreProcessing:  
    - `encode` Encodes class labels into one-hot or class codes.  
    - `decode` Decodes one-hot or class codes to class labels.  
    - `normalize` Scales data between 0 and 1.  
    - `split` Splits the data into training and testing sets.  

Training Callbacks:  
    - `early_stoppage` Stops the model if it hasn't improved in x epochs.  
    - `save_weights` Only save the best weights that model recorded.
  
Class `LoadModel` Loads a saved model.  

`nnet.py` contains the class for the Neural Network (NN) and other class functions.

Accuracy:  
##### 97% on 1st epoch. 98% after 3 epochs.

## Usage

#### main.py
```python
import numpy as np
from nnet import NN, LoadModel, PreProcessing
import matplotlib.pyplot as plt
# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data:
    X, Y = data['X'], data['Y']
# === PRE-PROCESSING ===
preprocess = PreProcessing()
X = X.reshape(X.shape[0], -1)
X = preprocess.normalize(X)
Y = preprocess.encode(Y, one_hot=False)
X_train, X_test, Y_train, Y_test = preprocess.split(X, Y)
del X, Y
# === NEURAL NETWORK ===
model = NN(verbose=True)
model.input(input_size=X_train.shape[1])
model.hidden(neurons=512, activation='relu')
model.hidden(neurons=512, activation='relu')
model.output(output_size=10, activation='softmax')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', learn_rate=0.1)
model.train(X_train, Y_train, batch_size=32, epochs=15, valid_split=0.1, early_stopping=5, save_weights=True)
model.evaluate(X_test, Y_test)
model.plot()
# === SAVE & LOAD ===
model.save('mnist')
mnist = LoadModel('mnist')
# === PLOT PREDICTION ===
rnum = np.random.randint(0, X_test.shape[0])
prediction, acc = model.predict(X_test[rnum])
print(f'Model: {preprocess.decode(prediction)} ({acc:.2%}) | Actual: {preprocess.decode(Y_test[rnum])}')
img_dims = int(np.sqrt(X_test.shape[1]))
plt.imshow(X_test[rnum].reshape(img_dims, img_dims), cmap='bone_r')
plt.show()
