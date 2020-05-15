import numpy as np
import warnings, dill, math, itertools
# import tensorflow as tf

class NNet:

    def __init__(self, verbose=False):
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")
        self.verbose = verbose
        self.layers = 0
        self.params = 0
        self.activations = []
        self.weights = []
        self.biases = []
        self.inputExists = False
        self.outputExists = False
        self.activators = ['relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu']
        self.optimizers = ["adam", "sgd", "rmsprop", "adadelta"]
        self.loss_functions = ['mae', 'mse', 'cce', 'scce', 'bce']
        self.scorers = ['accuracy']
        if self.verbose:
            print("* Neural Network Initialised *\n")

    @staticmethod
    def encode(target):
        if str(target.dtype)[:2] != '<U':
            raise TypeError(f"Cannot encode: {target.dtype} must be string/unicode format")
        output_size = len(np.unique(target))
        count = 0
        output = []
        output_dict = {}
        total = set()
        code = {}
        for i in target:
            if i in output_dict:
                output.append(output_dict[i])
            else:
                output_dict[i] = count
                output.append(output_dict[i])
                count += 1
                code[len(total)] = i
                total.add(i)
        return np.eye(output_size)[output], code

    @staticmethod
    def normalize(data):
            xmax, xmin = np.amax(data), np.amin(data)
            normalize = lambda x: (x - xmin) / (xmax - xmin)
            return normalize(data)

    @staticmethod
    def standardize(data):
            xmean, xstd = np.mean(data), np.std(data)
            standardize = lambda x: (x - xmean) / xstd
            return standardize(data)

    @staticmethod
    def split(data, target, test_split=1/5, shuffle=True, seed=None):
        if data is None:
            raise ValueError("X is None")
        if target is None:
            raise ValueError("Y is None")
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try:
                target = np.array(target)
            except:
                raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False only")
        if data.shape[0] != target.shape[0]:
            raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")
        samples = data.shape[0]
        if test_split>0 and test_split <1:
            if shuffle:
                rp = np.random.RandomState(seed=seed).permutation(samples)
                data, target = data[rp], target[rp]
            test_size = int(samples * test_split)
            train_size = samples - test_size
            assert (test_size + train_size) == samples
            return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]
        else:
            raise ValueError(f"Test split {test_split} must be a value between (excluding) 0 and 1")

    def input(self, input_size):
        if not isinstance(input_size, tuple):
            raise TypeError("Input size must be a tuple")
        if not input_size:
            raise ValueError("Input size is none")
        if len(input_size) > 2:
            self.flatten = True
            if not (input_size[1] == input_size[2]):
                raise ValueError(f"Image dimensions must be equal ({input_size[1]} vs. {input_size[2]})")
            self.input_size = input_size[1]**2
            if len(input_size) > 3:
                raise ValueError(f"Bad input shape: {input_size}")
        else:
            self.flatten = False
            self.input_size = input_size[1]

        self.inputExists = True

        if self.verbose:
            if self.flatten:
                print(f"Flattened:\t({int(self.input_size**0.5)}, {int(self.input_size**0.5)}) -> {self.input_size}")

    def hidden(self, neurons=128, activation="relu", dropout=False):
        if not self.inputExists:
            raise NotImplementedError("No input layer implemented yet - try input() first")
        if neurons<1 or neurons>1024 or not isinstance(neurons, int):
            raise ValueError(f"{neurons} must be an integer between 1 and 1024")
        if activation not in self.activators:
            raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")
        else:
            self.activations.append(activation)
        if not isinstance(dropout, bool):
            raise TypeError("Dropout must be True or False")

        self.layers += 1

        # get amount of input neurons needed for this layer from the previous layer's output size
        if self.layers == 1:
            # FIRST LAYER ONLY
            input_neurons = self.input_size
            if self.verbose:
                print("----------------------")
                print(f"\tInput [{self.input_size}]")
        else:
            # ALL OTHER LAYERS
            input_neurons = self.previous_output_size

        # parameter counter
        self.params += (input_neurons * neurons) + neurons
        # INITIALIZE WEIGHTS
        self.weights.append(np.random.randn(input_neurons, neurons) * np.sqrt(1/input_neurons))
        # INITIALIZE BIASES
        self.biases.append(np.random.randn(1, neurons))

        # set this layer's output neurons for the next layer
        self.previous_output_size = neurons

        if self.verbose:
            if dropout:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout")
            else:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, output_size, activation=None):
        if self.layers == 0:
            raise NotImplementedError("No hidden layer(s) implemented yet - try hidden() first")
        if activation not in self.activators:
            raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")
        else:
            self.activations.append(activation)

        # set output size to the shape[1] of the target data
        try:
            self.output_size = output_size[1]
        except:
            self.output_size = 1

        self.outputExists = True

        # parameter counter 
        self.params += (self.previous_output_size * self.output_size) + self.output_size
        self.weights.append(np.random.randn(self.previous_output_size, self.output_size) * np.sqrt(1/self.previous_output_size))
        self.biases.append(np.random.randn(1, self.output_size))
        
        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})")
            print("----------------------")
            print(f"Parameters:\t{self.params:,}")

    def compile(
        self, valid_split=1/10, optimizer="adam", loss="cce",
        scorer="accuracy", learn_rate=1e-3, batch_size=64, epochs=15):

        if not self.outputExists:
            raise NotImplementedError("No output layer implemented yet - try output() first")
        if optimizer not in self.optimizers:
            raise ValueError(f"{optimizer} is not a valid optimizer. Options: {self.optimizers}")
        else:
            self.optimizer = optimizer
        if loss not in self.loss_functions:
            raise ValueError(f"{loss} is not a valid loss function. Options: {self.loss_functions}")
        else:
            self.loss = loss
        if scorer not in self.scorers:
            raise ValueError(f"{scorer} is not a valid scorer. Options: {self.scorers}")
        else:
            self.scorer = scorer

        if batch_size>0 and isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError(f"Batch size {batch_size} must be an integer greater than 1")
        if batch_size > 128:
            warnings.warn("Batch sizes greater than 128 are not recommended")
        if epochs > 0 and epochs < 1000 and isinstance(epochs, int):
            self.epochs = epochs
        else:
            raise ValueError(f"({epochs}) Number of epochs must be an integer between 1 and 999")

        self.learn_rate = learn_rate

        if valid_split>0 and valid_split <1:
            self.valid_split = valid_split
        else:
            raise ValueError(f"({valid_split}) Validation split must be a value between 0 and 1")

        if self.verbose:
            keys = locals()
            del keys['self']
            print(keys)

    @staticmethod
    def Activation(act, x, d=False):
        if act == 'sigmoid':
            if d:
               return (1.0/(1.0 + np.exp(-x))) * (1-(1.0/(1.0 + np.exp(-x))))
            else:
                return 1.0/(1.0 + np.exp(-x))
        if act == 'tanh':
            if d:
                return 1.0 - np.tanh(x)**2
            else:
                return np.tanh(x)
        if act == 'relu':
            if d:
                return 1 if x > 0 else 0
            else:
                return np.maximum(0,x)
        if act == 'leaky_relu':
            if d:
                return 1 if x > 0 else .01
            else:
                return np.maximum(.01*x, x)
        if act == 'softmax':
            exps = np.exp(x - x.max())
            #return np.exp(x) / np.sum(np.exp(x))
            return exps / np.sum(exps, axis=0)

    @staticmethod
    def Loss(loss, y_pred, y_true):
        if loss == 'cce':
            losses = []
            for pred, label in zip(y_pred, y_true):
                pred /= pred.sum(axis=-1, keepdims=True)
                pred = np.clip(pred, 1e-7, 1 - 1e-7)
                losses.append(np.sum(label * -np.log(pred), axis=-1, keepdims=False))
            return np.mean(losses)
        if loss == 'mae': #L1
            return np.sum(np.absolute(y_pred - y_true))
        if loss == 'mse': #L2
            return np.sum((y_pred - y_true)**2) / y_true.size

    def train(self, data, target):
        if data is None:
            raise ValueError("X is None")
        if target is None:
            raise ValueError("Y is None")
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try:
                target = np.array(target)
            except:
                raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")
        if data.shape[0] != target.shape[0]:
            raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")
        # flatten data if applicable before being trained
        if self.flatten:
            data = data.reshape(data.shape[0], -1)

        # split into training and validation data
        self.valid_size = int(data.shape[0] * self.valid_split)
        self.train_size = data.shape[0] - self.valid_size
        assert (self.train_size + self.valid_size) == data.shape[0]
        X_train, self.X_valid = (data[:self.train_size], data[-self.valid_size:])
        Y_train, self.Y_valid = (target[:self.train_size], target[-self.valid_size:])

        # reset batch size if greater than training samples
        if self.batch_size >= self.train_size:
            self.batch_size = self.train_size

        # if self.verbose:
        print(f'\nTraining on {self.train_size:,} samples...')

        #loop for each epoch
        for i in range(self.epochs):
            # batch stuff
            sbatch = 0
            ebatch = self.batch_size
            batch_finished = False
            is_last_run = False
            batches = int(math.ceil(self.train_size/self.batch_size))
            count = 1

            # loop through batches
            while not batch_finished:
                # print the epoch, batch and sample info
                #set batches
                xtrain, ytrain = X_train[sbatch:ebatch], Y_train[sbatch:ebatch]

                # get output of a batch forward pass
                output = self.forward(xtrain)

                # calculate loss from the output of the forward pass to ytrain
                loss = self.Loss(self.loss, output, ytrain)
                # cce = tf.keras.losses.CategoricalCrossentropy()
                # loss2 = cce(ytrain, output)
                # print(f"({loss2}")
                # gradient descent (backwards pass here)
                print(f"Epoch {i+1}/{self.epochs}\tSamples {ebatch}/{self.train_size}\tTrain Loss {loss:4f}", end='\r')
                self.backProp(loss, output, ytrain)

                # move to the next batch
                sbatch = ebatch
                # if the next batch will go equal or after the training size
                if ebatch + self.batch_size >= self.train_size:
                    ebatch = self.train_size
                    # end the batch loop
                    if is_last_run:
                        batch_finished = True
                    # set the last batch run
                    is_last_run = True
                else:
                    # increase the end batch num by batch size
                    ebatch += self.batch_size

                count += 1

            # validate the newly optimized weights and biases
            valid_loss, valid_acc = self.validate()

            #print loss and accuracy after each epoch
            print(f"\nVal Loss: {loss:.4f} | Val Accuracy: {valid_acc:.3%}")
            # self.output = np.concatenate(output_list)

    def forward(self, X):

        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            X = np.dot(X, weight) + bias
            X = self.Activation(activation, X)
        return X

    def backProp(self, loss, output, ytrain):

        pass
        # cost = np.mean(ytrain-output)
        # print(cost)
        # delta = self.Activation(self.activations[-1], loss, d=True) * cost
        # for index in range(len(self.weights)-1, 0, -1):
        #     delta = np.dot(self.weights[index].T, delta) * self.Activation(self.activations[index-1], loss, d=True)
        #     delta_bias = delta
        #     self.weights[index] -= self.learn_rate * self.gradient(delta, self.activations[index])
        #     self.biases[index] -= self.learn_rate * self.gradient(delta_bias, self.activations[index])

    def gradient(self, x, act):
        return np.dot(x.T, self.Activation(act, x, d=True))

    def validate(self):
        
        predictions = self.forward(self.X_valid)

        valid_loss = self.Loss(self.loss, predictions, self.Y_valid)

        if self.output_size > 1:
            targets = np.argmax(self.Y_valid, axis=1)
            predictions = np.argmax(predictions, axis=1)
        else:
            targets = self.Y_valid

        correct = np.sum(predictions == targets)
        valid_acc =  correct/self.valid_size
        return valid_loss, valid_acc

    # Automatic predictions and evaluations based on test data
    def evaluate(self, data, target):
        if data is None:
            raise ValueError("X is None")
        if target is None:
            raise ValueError("Y is None")
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try:
                target = np.array(target)
            except:
                raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")
        if data.shape[0] != target.shape[0]:
            raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")
        # flatten data if applicable before being trained
        if self.flatten:
            data = data.reshape(data.shape[0], -1)

        print(f"\nTesting Set Evaluation:")
        print("-----------------------")

        predictions = self.forward(data)

        loss = self.Loss(self.loss, predictions, target)

        if self.output_size > 1:
            targets = np.argmax(target, axis=1)
            predictions = np.argmax(predictions, axis=1)
        else:
            targets = target

        self.correct = np.sum(predictions == targets)
        self.test_acc = self.correct/target.shape[0]
        print(f"Test Acc:\t{self.correct}/{target.shape[0]} ({self.test_acc:.2%})")
        print(f"Test Loss:\t{loss:.4f}\n")

    def save(self, file):
        path = 'models/' + file + '.pkl'
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print(f"'{file.upper()}' model saved.")

    @staticmethod
    def load(file):
        path = 'models/' + file + '.pkl'
        with open(path, 'rb') as f:
            print(f"'{file.upper()}' model loaded.\n")
            return dill.load(f)

    def predict(self, data):
        if data is None:
            raise ValueError("X is None")
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if self.flatten:
            if len(data.shape) == 2:
                data = data.flatten()
            else:
                data = data.reshape(data.shape[0], -1)

        output = self.forward(data)

        if self.output_size > 1:
            if output.shape[0] > 1:
                prediction = np.argmax(output, axis=1)
            else:
                prediction = np.argmax(output)
            score = np.amax(output)
            return prediction, score
        else:
            return prediction
