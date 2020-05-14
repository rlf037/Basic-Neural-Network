import numpy as np
import warnings
import dill

class NN:
    def __init__(self, verbose=True):

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")

        self.verbose = verbose
        self.flattened = False
        self.inputExists = False
        self.splitted = False
        self.transformed = False
        self.transforms = []
        self.activations = []
        self.hlayers = 0
        self.params_count = 0
        self.weights = []
        self.biases = []
        self.outputExists = False

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
            return lambda y1, y2: np.max(0, 1 - y2 * y1)

    @staticmethod
    def scce(y1, y2):
        if y1 == 1:
            return lambda y1, y2: -log(y2)
        else:
            return lambda y1, y2: -log(1 - y2)

    def encode(self, target):
        self.output_size = len(np.unique(target))
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
        self.Y = np.eye(self.output_size)[output]

    def input(self, data=None, target=None, flatten=False, problem=None):
        if data is None:
            raise ValueError("No X data passed. Use data=")
        if target is None:
            raise ValueError("No Y data passed. Use target=")
        if not isinstance(data, np.ndarray):
            raise TypeError(f"X of type {type(data)} must be a NumPy array")
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Y of type {type(target)} must be a NumPy array")

        if not problem:
            if str(target.dtype)[:2] == '<U':
                problem = 'classification'
            else:
                problem = 'regression'
        else:
            if problem not in ["classification", "regression"]:
                raise ValueError(
                    f"Only 'classification' or 'regression' are valid problems"
                )

        if data.shape[0] != target.shape[0]:
            raise ValueError(
                f"X and Y sample sizes are not equal ({data.shape[0]} vs. {target.shape[0]})"
            )

        if flatten == True:
            data = data.reshape(data.shape[0], -1)
            self.flattened = True
        else:
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
                self.flattened = True

        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], -1)

        if len(data.shape) != 2:
            raise ValueError(
                f"Bad X input shape {data.shape}. Try flattening the data with flatten=True"
            )

        if problem == "classification":
            self.problem = "classification"
            self.encode(target)
        else:
            self.problem = "regression"
            self.output_size = 1
            self.Y = target

        self.target = target
        self.inputExists = True
        self.X = data
        self.input_size = data.shape[1]
        self.samples = data.shape[0]

        if self.verbose:
            print(f"Problem:\t{self.problem.capitalize()}")
            if self.flattened:
                print(
                    f"Flattened:\t({int(self.input_size**0.5)}, {int(self.input_size**0.5)}) -> {self.input_size}"
                )
            print(f"Samples:\t{self.samples:,}")
            print(f"Features:\t{self.input_size:,}")
            if self.problem == "classification":
                print(f"Classes:\t{self.output_size}")
                print(f"Encoding:\tOne-Hot")

    def transform(self, transform):
        trans = ["normalize", "standardize"]

        if not self.inputExists:
            raise NotImplementedError( "No input detected. Use input()")

        if self.splitted:
            raise RuntimeError(
                "Transforms must be done before splitting the data to ensure the same transformations are applied to both the training set and testing set"
            )
        if transform not in trans:
            raise ValueError(
                f"'{transform}' is not a valid transform. Options: {trans}"
            )

        if transform == "normalize":
            xmax, xmin = np.amax(self.X), np.amin(self.X)
            self.normalize = lambda x: (x - xmin) / (xmax - xmin)
            self.X = self.normalize(self.X)

        if transform == "standardize":
            xmean, xstd = np.mean(self.X), np.std(self.X)
            self.standardize = lambda x: (x - xmean) / xstd
            self.X = self.standardize(self.X)

        self.transformed = True
        self.transforms.append(transform)

        if self.verbose:
            print(f"Transform:\t{self.transforms[0].capitalize()[:-1]}ation")

    def split(self, test_split=1/10, shuffle=True, seed=None):
        if not self.inputExists:
            raise NotImplementedError( "No input detected. Use input()")

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False only")

        if shuffle:
            r = np.random.RandomState(seed=seed).permutation(self.samples)
            self.X, self.Y, self.target = self.X[r], self.Y[r], self.target[r]

        try:
            self.test_size = int(self.samples * test_split)
            self.X_train, self.X_test = (
                self.X[: -self.test_size],
                self.X[-self.test_size :],
            )
            self.Y_train, self.Y_test = (
                self.Y[: -self.test_size],
                self.Y[-self.test_size :],
            )
            self.target = self.target[: -self.test_size]
            self.splitted = True
            self.train_size = self.X_train.shape[0]
            del self.X, self.Y
        except:
            raise ValueError(f"{test_split} is not a valid test split. Default is 1/7")

        if self.verbose:
            print(
                f"Train:\t\t{int(self.train_size):,} ({1.0-test_split:.2%})"
            )
            print(
                f"Test:\t\t{int(self.X_test.shape[0]):,} ({test_split:.2%})"
            )
            print(f"Shuffled:\t{shuffle}")
            print("----------------------")

    def addLayer(self, neurons=128, activation="relu", dropout=False):
        self.acts = ["relu", "tanh", "sigmoid", "softmax"]
        if not self.splitted:
            raise NotImplementedError( "You have not split the data into training and test sets yet. Use split()")

        if neurons < 1 or neurons > 1024:
            raise ValueError(
                f"Number of neurons must be between 1 and 1024 not {neurons}"
            )

        if activation not in self.acts:
            raise ValueError(
                f"{activation} is not a valid activation function. Options: {self.acts}"
            )
        else:
            self.activations.append(activation)

        if not isinstance(dropout, bool):
            raise TypeError(
                "dropout must be True or False only"
            )

        self.hlayers += 1

        if self.hlayers == 1:
            self.n_input_size = self.input_size
            print(f"\tInput [{self.X_train.shape[1]}]")
        else:
            self.n_input_size = self.previous_layer_size

        self.params_count += self.n_input_size * neurons
        self.weights.append(np.random.standard_normal() * np.random.randn(self.n_input_size, neurons))
        self.biases.append(np.zeros(neurons))
        self.params_count += neurons
        self.previous_layer_size = neurons

        if self.verbose:
            if dropout:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout")
            else:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, activation="softmax"):

        if self.hlayers < 1:
            raise NotImplementedError(
                "No hidden layers detected. Try addLayer()"
            )

        if activation not in self.acts:
            raise ValueError(
                f"{activation} is not a valid activation function. Options: {self.acts}"
            )

        self.outputExists = True
        self.params_count += self.previous_layer_size * self.output_size
        self.output_weights = np.random.standard_normal() * np.random.randn(
            self.previous_layer_size, self.output_size
        )
        self.output_biases = np.zeros(self.output_size)
        self.params_count += self.output_size

        self.output_act = activation

        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})")
            if self.problem == 'regression' and activation == 'softmax':
                warnings.warn("Softmax is not recommended for regression. Try a linear function")
            print("----------------------")
            print(f"Parameters:\t{self.params_count:,}")

    def compile(
        self, valid_split=1/10, optimizer="adam", loss="mse", scorer="accuracy", learn_rate=0.001):
        if not self.outputExists:
            raise NotImplementedError("Model has no output yet. Use output()")
        opts = ["adam", "sgd", "rmsprop", "adadelta"]
        losses = ['mae', 'mse', 'cce']
        scorers = ['accuracy']

        if optimizer not in opts:
            raise ValueError(
                f"{optimizer} is not a valid optimizer. Options: {opts}"
            )
        if loss not in losses:
            raise ValueError(
                f"{loss} is not a valid loss function. Options: {losses}"
            )
        if loss != 'cce' and self.problem == 'classification':
            warnings.warn("Categorical Cross Entropy ('cce') is recommended as the loss function when doing classification")
        if loss != 'cce' and self.problem == 'regression':
            warnings.warn("'cce' is not recommended as the loss function for regression. Try 'mse' or 'mae'")
        if scorer not in scorers:
            raise ValueError(
                f"{scorer} is not a valid scorer. Options: {scorers}"
            ) 

        try:
            self.valid_size = int(self.X_train.shape[0] * valid_split)
            self.X_train, self.X_valid = (
                self.X_train[: -self.valid_size],
                self.X_train[-self.valid_size :],
            )
            self.Y_train, self.Y_valid = (
                self.Y_train[: -self.valid_size],
                self.Y_train[-self.valid_size :],
            )
        except:
            raise ValueError(f"{valid_split} is not a valid validation split. Default is 1/10")

        if self.verbose:
            print(f"Validation:\t{valid_split:.2%}")
            print(f"Loss Func:\t{loss.upper()}")
            print(f"Optimizer:\t{optimizer.capitalize()}")
            print(f"Scorer:\t\t{scorer.capitalize()}")

    def train(self, batch_size=32, epochs=10):

        if batch_size > 0 and batch_size < self.train_size and isinstance(batch_size, int):
            if self.verbose:
                print(f"Batch Size:\t{batch_size}")
            self.batch_size = batch_size
        else:
            raise ValueError(
                f"batch_size {batch_size} must be an integer between 1 and {self.train_size}"
            )

        if batch_size > 999:
            warnings.warn("Batch sizes greater than 999 are not recommended.")

        if epochs > 0 and epochs < 1000 and isinstance(epochs, int):
            if self.verbose:
                print(f"Epochs:\t\t{epochs}")
            self.epochs = epochs
        else:
            raise ValueError(
                f"epochs {epochs} must be an integer between 1 and 999"
            )

        if self.verbose:
            print('\nStarted training...')

        # 1 hidden layer only
        if self.hlayers == 1:
            self.last_output = np.dot(self.X_train, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            self.last_output = act_func(self.last_output)
        # multiple hidden layers
        else:
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    self.X_train = self.last_output

                self.last_output = (
                    np.dot(self.X_train, self.weights[layer_num])
                    + self.biases[layer_num]
                )
                act_func = self.actFunc(self.activations[layer_num])
                self.last_output = act_func(self.last_output)

        self.output = np.dot(self.last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)
        self.output = act_func(self.output)

        if self.verbose:
            print('Finished training!\n')
        
    # Automatic predictions and evaluations based on test data
    def evaluate(self):
        #convert from one-hot encoding back to category integers and then decode back to original class
        # with self.code dictionary that stored each class as a value to the integer key
        if self.problem == "classification":
            self.predictions = np.argmax(self.output, axis=1)
            self.predictions = [self.code[x] for x in self.predictions]
            self.predictions = np.array(self.predictions)
        else:
            self.predictions = self.output

        if self.verbose:
            if self.problem == 'classification':
                print(f"{self.Y_test.shape} -> {self.predictions.shape} -> {self.predictions.shape} | {self.output_act.capitalize()} -> ArgMax -> Decode")

        print(f"Predicting on {self.test_size:,} samples...")
        print("-------------------------------")

        self.target = self.target[: -self.valid_size]
        self.correct = np.sum(self.predictions == self.target)
        self.test_acc =  self.correct/self.target.shape[0]
        print(f"Accuracy: {self.correct}/{self.target.shape[0]} ({self.test_acc:.2%})\n")

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
        self.last_output = 0
        # Perform same flattening (if applicable) and transformations to outside data to predict with:
        if self.flattened:
            data = data.reshape(-1)

        if self.transformed:
            if self.transforms[0] == "normalize":
                data = self.normalize(data)

            if self.transforms[0] == "standardize":
                data = self.standardize(data)

        # 1 hidden layer only
        if self.hlayers == 1:
            last_output = np.dot(data, self.weights[0]) + self.biases[0]
            act_func = self.actFunc(self.activations[0])
            last_output = act_func(last_output)
        # multiple hidden layers
        else:
            for layer_num in range(len(self.weights)):
                # skip on first layer iteration (last_output not set yet)
                if layer_num != 0:
                    data = last_output

                last_output = (
                    np.dot(data, self.weights[layer_num])
                    + self.biases[layer_num]
                )
                act_func = self.actFunc(self.activations[layer_num])
                last_output = act_func(last_output)

        output = np.dot(last_output, self.output_weights) + self.output_biases
        act_func = self.actFunc(self.output_act)
        output = act_func(output)

        if self.problem == "classification":
                prediction = np.argmax(output)
                prediction = self.code[prediction]
        else:
            prediction = output

        return prediction


