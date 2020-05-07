# Basic Neural Network

### v1.11

A simple Neural Network written in Python.

NeuralNetwork.py contains the class for the Neural Network. It is a much simplier implmentation in that data preprocessing/transformation and training split can all be done in the model definition.

If you pass the training and test data to the model at the beginning, it can use that data to infer parameters so this is a quicker implementation. You can see pass different testing data if need be or use your own strict parameters. Eventually the model will run with default parameters, only needing passed X/Y data the model definition.

## Usage

```python
nn = NeuralNetwork()

nn.input(data=X, labels=Y)
nn.transform(Y=True, transform='categorical') #labels transform
nn.transform(X=True, transform='normalize') #data transform
nn.split() #split data into train/test samples before training
nn.addLayer()
nn.addLayer()
nn.addLayer()
nn.output()

nn.compile()
nn.summary()
nn.train()
nn.predict()
nn.evaluate() 
```