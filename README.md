# Basic Neural Network

### v1.18

A simple Neural Network written in Python without the use of external libraries (except NumPy).

`neuralnetwork.py` contains the class for the Neural Network (NN). It is an easy to use implementation in that data preprocessing/transformation, train/test split and retransformations can all be done in the same object which gives a better linear picture of the network.

All that's required is to pass X and Y to the model input and the rest of the parameters can be derived automatically based on the inputted data. Automatic parameters are input size, flattening, problem (classification or regression), transformations, train/test split, layer size, layer activations, dropout(s), output activation, output size, validation split, optimizers, scorers, early stoppages, batch sizes, epochs and retransformation. One-hot encoding is automatically applied for classification and translated back on prediction.

Warnings are also in-built if the user selects paramaters that are not optimised/recommended depending on the dataset type.

## Usage

```python
from neuralnetwork import NN

nn = NN(verbose=True)
nn.input(data=X, target=Y)
nn.transform("normalize")
nn.split(test_split=1/7, shuffle=True)
nn.addLayer(neurons=10, activation="relu", dropout=True)
nn.addLayer(neurons=15, activation="sigmoid", dropout=False)
nn.addLayer(neurons=5, activation="tanh", dropout=False)
nn.output(activation="softmax")
nn.compile(valid_split=1/10, loss='cce', optimizer="adam", scorer="accuracy", learn_rate=0.001)
nn.train(batch_size=32, epochs=10)
nn.predict()
nn.evaluate()
```
