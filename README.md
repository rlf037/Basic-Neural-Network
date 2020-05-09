# Basic Neural Network

### v1.14

A simple Neural Network written in Python without the use of external libraries (except NumPy).

NeuralNetwork.py contains the class for the Neural Network. It is a much simplier implementation in that data preprocessing/transformation, train/test split and retransformations can all be done in the same object which gives a better linear picture of the network.

All that's required is to pass X and Y to the model input and the rest of the parameters can be derived automatically based on the passed data. Automatic parameters are input size, flattening, problem (classification or regression), transformations, train/test split, layer size, layer activations, dropout(s), output activation, output size, validation split, optimizers, scorers, early stoppages, batch sizes, epochs and retransformation.

## Usage

```python
from neuralnetwork import NeuralNetwork

nn = NeuralNetwork(verbose=True)
nn.input(data=X, target=Y, flatten=True, problem='classification')
nn.transform('normalize')
nn.split(test_split=1/7, shuffle=True)
nn.addLayer(neurons=128, activation='relu', dropout=False)
nn.addLayer(neurons=64, activation='relu', dropout=True)
nn.addLayer(neurons=16, activation='relu', dropout=False)
nn.output(activation='softmax')
nn.compile(valid_split=0.2, optimizer='adam', scorer='accuracy', early_stoppage=True)
nn.train(batch_size=32, epochs=10)
nn.predict('argmax')
nn.evaluate() 
```
