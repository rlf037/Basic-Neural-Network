# Basic Neural Network

A simple Neural Network written in Python.

NeuralNetwork.py contains the class for the Neural Network. It is a much simplier implmentation in that data preprocessing/transformation and training split can all be done in the model definition.

## Usage

eg.
```python
model = NeuralNetwork()

model.add(layer='input', data=X_train, labels=Y_train)
model.add(layer='transform', transform='categorical') #labels transform
model.add(layer='transform', transform='normalize') #data transform
model.add(layer='hidden', n_neurons=128)
model.add(layer='hidden', n_neurons=128)
model.add(layer='output')

model.summary()
model.compile()
model.fit()
model.predict()
model.evaluate() 
```