# Basic Neural Network

A simple Neural Network written in Python.

NeuralNetwork.py contains the class for the Neural Network. It is a much simplier implmentation in that data preprocessing/transformation and training split can all be done in the model definition.

If you pass the training and test data to the model at the beginning, it can use that data to infer parameters so this is a quicker implementation. You can see pass different testing data if need be or use your own strict parameters. Eventually the model will run with default parameters, only needing passed X/Y data the model definition.

## Usage

eg.
```python
model = NeuralNetwork()

model.input(data=X, labels=Y, flatten=True)
model.transform(Y=True, transform='categorical') #labels transform
model.transform(X=True, transform='normalize') #data transform
model.split(test_split=1/7, shuffle=True, random_state=0) #split data into train/test samples before training
model.add(n_neurons=500, activation='relu')
model.add(n_neurons=500, activation='relu')
model.output(activation='softmax')

model.compile(valid_split=0.2, optimizer='adam', batch_size=128, epochs=10)
model.summary()
model.train()
model.predict()
model.evaluate()
```