import numpy as np
import warnings
import dill
from math import log2

class NN:
    def __init__(self, verbose=False):

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")

        self.verbose = verbose
        self.activations = []
        self.layers = 0
        self.params = 0
        self.weights = []
        self.biases = []

        if self.verbose:
            print("* Neural Network Initialised *\n")

    @staticmethod
    def actFunc(act):
        if act == 'sigmoid':
            return lambda x: 1.0/(1.0 + np.exp(-x))
        if act == 'tanh':
            return lambda x: np.tanh(x)
        if act == 'relu':
            return lambda x: np.maximum(0,x)
        if act == 'softmax':
            return lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def lossFunc(loss):
        if loss == 'mae': #L1
            return lambda y1, y2: np.sum(np.absolute(y1 - y2))
        if loss == 'mse': #L2
            return lambda y1, y2: np.sum((y1 - y2)**2) / y2.size
        if loss == 'scce':
            return scce(y1, y2)
        if loss == 'cce':
            return lambda p, q: -sum([p[i]*log2(q[i]) for i in range(len(p))])

    @staticmethod
    def scce(y1, y2):
        if y1 == 1:
            return lambda y1, y2: -log(y2)
        else:
            return lambda y1, y2: -log(1 - y2)

    def encode(self, target):
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
        self.code = code
        return np.eye(output_size)[output]

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
            raise ValueError("No X data passed. Use data=")
        if target is None:
            raise ValueError("No Y data passed. Use target=")
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise TypeError(f"Cannot convert 'data' of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try:
                target = np.array(target)
            except:
                raise TypeError(f"Cannot convert 'target' of type {type(target)} to a NumPy array")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False only")
        if data.shape[0] != target.shape[0]:
            raise ValueError(f"data and target sample sizes are not equal ({data.shape[0]} vs. {target.shape[0]})")
        samples = data.shape[0]
        if test_split>0 and test_split <1:
            if shuffle:
                rp = np.random.RandomState(seed=seed).permutation(samples)
                data, target = data[rp], target[rp]
            test_size = int(samples * test_split)
            train_size = samples - test_size
            return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]
        else:
            raise ValueError(f"test_split {test_split} must be a value between 0 and 1")

    def input(self, input_size):
        if len(input_size) > 2:
            self.flatten = True
            self.input_size = input_size[1]**2
            if len(input_size) > 3:
                raise ValueError(f"Bad input shape {input_size}")
        else:
            self.flatten = False
            self.input_size = input_size[1]

        self.train_size = input_size[0]

        if self.verbose:
            if self.flatten:
                print(f"Flattened:\t({int(self.input_size**0.5)}, {int(self.input_size**0.5)}) -> {self.input_size}")

    def addLayer(self, neurons=128, activation="relu", dropout=False):
        # * CHECKS
        self.acts = ["relu", "tanh", "sigmoid", "softmax"]
        if neurons < 1 or neurons > 1024:
            raise ValueError(f"Number of neurons must be between 1 and 1024 not {neurons}")
        if activation not in self.acts:
            raise ValueError(f"{activation} is not a valid activation function. Options: {self.acts}")
        else:
            self.activations.append(activation)
        if not isinstance(dropout, bool):
            raise TypeError("dropout must be True or False only")
        # * END

        self.layers += 1

        if self.layers == 1:
            self.n_input_size = self.input_size
            if self.verbose:
                print("----------------------")
                print(f"\tInput [{self.input_size}]")
        else:
            self.n_input_size = self.previous_layer_size

        self.params += self.n_input_size * neurons
        self.weights.append(.01 * np.random.randn(self.n_input_size, neurons))
        self.biases.append(np.zeros(neurons))
        self.params += neurons
        self.previous_layer_size = neurons

        if self.verbose:
            if dropout:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout")
            else:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, output_size, activation="softmax"):
        if self.layers < 1:
            raise NotImplementedError("No hidden layers detected. Try addLayer()")
        if activation not in self.acts:
            raise ValueError(f"{activation} is not a valid activation function. Options: {self.acts}")
        try:
            self.output_size = output_size[1]
        except:
            self.output_size = 1
        self.outputExists = True
        self.params += self.previous_layer_size * self.output_size
        self.output_weights = .01 * np.random.randn(self.previous_layer_size, self.output_size)
        self.output_biases = np.zeros(self.output_size)
        self.params += self.output_size
        self.output_act = activation

        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})")
            print("----------------------")
            print(f"Parameters:\t{self.params:,}")

    def compile(
        self, valid_split=1/10, optimizer="adam", loss="cce",
        scorer="accuracy", learn_rate=1e-3, batch_size=64, epochs=15):

        if not self.outputExists:
            raise NotImplementedError("Model has no output yet. Use output()")
        opts = ["adam", "sgd", "rmsprop", "adadelta"]
        losses = ['mae', 'mse', 'cce', 'scce', 'bce']
        scorers = ['accuracy']
        if optimizer not in opts:
            raise ValueError(f"{optimizer} is not a valid optimizer. Options: {opts}")
        if loss not in losses:
            raise ValueError(f"{loss} is not a valid loss function. Options: {losses}")
        if scorer not in scorers:
            raise ValueError(f"{scorer} is not a valid scorer. Options: {scorers}")
        if batch_size > 0 and batch_size < self.train_size and isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError(f"batch_size {batch_size} must be an integer between 1 and {self.train_size}")
        if batch_size > 999:
            warnings.warn("Batch sizes greater than 999 are not recommended.")
        if epochs > 0 and epochs < 1000 and isinstance(epochs, int):
            self.epochs = epochs
        else:
            raise ValueError(f"epochs {epochs} must be an integer between 1 and 999")

        self.loss = loss
        self.learn_rate = learn_rate

        if valid_split>0 and valid_split <1:
            self.valid_split = valid_split
        else:
            raise ValueError(f"valid_split {valid_split} must be a value between 0 and 1")

        if self.verbose:
            print(f"Optimizer:\t{optimizer.capitalize()}")
            print(f"Scorer:\t\t{scorer.capitalize()}")
            if self.loss == 'mae':
                a = 'Mean Absolute Error'
            if self.loss == 'mse':
                a = 'Mean Squared Error'
            if self.loss == 'cce':
                a = 'Categorical Crossentropy'
            if self.loss == 'scce':
                a = 'Sparse Categorical Crossentropy'
            if self.loss == 'bce':
                a = 'Binary Crossentropy'
            print(f"Loss:\t\t{a}")
            print(f"Learn Rate:\t{self.learn_rate}")

    def train(self, data, target):

        if self.flatten:
            data = data.reshape(self.train_size, -1)

        self.valid_size = int(self.train_size * self.valid_split)
        self.train_size -= self.valid_size
        self.X_train, self.X_valid = (data[:self.train_size], data[-self.valid_size:])
        self.Y_train, self.Y_valid = (target[:self.train_size], target[-self.valid_size:])

        if self.verbose:
            print('\nTraining...')

        #loop for each epoch
        for i in range(self.epochs):
            print(f"Epoch\t{i+1}/{self.epochs}:\t", end='')

            sbatch = 0
            ebatch = self.batch_size
            batch_finished = False
            is_last_run = False
            output_list = []

            # loop through batches
            while not batch_finished:
                xtrain, ytrain = self.X_train[sbatch:ebatch], self.Y_train[sbatch:ebatch]
                # get output of a batch forward pass
                output = self.feedForward(xtrain)
                output_list.append(output)
                # get the loss function
                loss_func = self.lossFunc(self.loss)
                # calculate loss from the output of the forward pass
                # print(output[0])
                # print(ytrain[0])
                loss_list = []
                # print(ytrain[i])
                # print(output[0])
                for j in range(len(ytrain)):
                    loss_list.append(loss_func(ytrain[j], output[j]))
                
                loss = np.mean(loss_list)
                # print(loss)
                
                # gradient descent (backwards pass here)
                self.backProp(loss)

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

            valid = self.validate()
            print(f"Loss: {loss:.5} | Accuracy: {valid:.3%}")
            self.output = np.concatenate(output_list)

    def feedForward(self, X):
        # 1 hidden layer only
        if self.layers == 1:
            last_output = np.dot(X, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            last_output = act_func(last_output)
        # multiple hidden layers
        else:
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    X = last_output

                last_output = (np.dot(X, self.weights[layer_num])+ self.biases[layer_num])
                act_func = self.actFunc(self.activations[layer_num])
                last_output = act_func(last_output)

        output = np.dot(last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)

        return act_func(output)

    def backProp(self, loss):
        pass
        
    def validate(self):
        data = self.X_valid
        # 1 hidden layer only
        if self.layers == 1:
            last_output = np.dot(self.X_valid, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            last_output = act_func(last_output)
        else:
            # multiple hidden layers
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    data = last_output
                last_output = (np.dot(data, self.weights[layer_num]) + self.biases[layer_num])
                act_func = self.actFunc(self.activations[layer_num])
                last_output = act_func(last_output)

        output = np.dot(last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)
        output = act_func(output)

        if self.output_size > 1:
            targets = np.argmax(self.Y_valid, axis=1)
            predictions = np.argmax(output, axis=1)
        else:
            predictions = output
            targets = self.Y_valid

        correct = np.sum(predictions == targets)
        valid_acc =  correct/self.valid_size
        return valid_acc

    # Automatic predictions and evaluations based on test data
    def evaluate(self, data, target):
        #convert from one-hot encoding back to category integers and then decode back to original class
        # with self.code dictionary that stored each class as a value to the integer key
        if self.flatten:
            data = data.reshape(data.shape[0], -1)

        # 1 hidden layer only
        if self.layers == 1:
            last_output = np.dot(data, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            last_output = act_func(last_output)
        else:
            # multiple hidden layers
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    data = last_output
                last_output = (np.dot(data, self.weights[layer_num]) + self.biases[layer_num])
                act_func = self.actFunc(self.activations[layer_num])
                last_output = act_func(last_output)

        output = np.dot(last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)
        output = act_func(output)

        if self.output_size > 1:
            targets = np.argmax(target, axis=1)
            predictions = np.argmax(output, axis=1)
        else:
            predictions = output

        print(f"\nTesting Set Evaluation:")
        print("-----------------------")

        self.correct = np.sum(predictions == targets)
        self.test_acc = self.correct/target.shape[0]
        print(f"Accuracy: {self.correct}/{target.shape[0]} ({self.test_acc:.2%})\n")

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
        # Perform same flattening (if applicable) and transformations to outside data to predict with:
        if self.flatten:
            data = data.reshape(-1)

        # 1 hidden layer only
        if self.layers == 1:
            last_output = np.dot(data, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            last_output = act_func(last_output)
        # multiple hidden layers
        else:
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    data = last_output

                last_output = (np.dot(data, self.weights[layer_num]) + self.biases[layer_num])
                act_func = self.actFunc(self.activations[layer_num])
                last_output = act_func(last_output)

        output = np.dot(last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)
        output = act_func(output)

        if self.output_size > 1:
            prediction = np.argmax(output)
            prediction = self.code[prediction]
            score = np.amax(output)
            return prediction, score, self.code
        else:
            return output
