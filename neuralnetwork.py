import numpy as np


class NeuralNetwork:
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

    def input(self, data=None, target=None, flatten=False, problem=None):
        if data is None:
            raise ValueError("No X data passed. Use data=")
        if target is None:
            raise ValueError("No Y data passed. Use target=")
        if not isinstance(data, np.ndarray):
            raise TypeError(f"X of type {type(data)} must be a NumPy array")
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Y of type {type(target)} must be a NumPy array")

        if problem not in ["classification", "regression"]:
            raise ValueError(
                f"Missing problem='classification' or problem='regression'"
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

        if len(data.shape) != 2:
            raise ValueError(
                f"Bad X input shape {data.shape}. Try flattening the data with flatten=True"
            )

        if problem == "classification":
            self.problem = "classification"
            self.output_size = len(np.unique(target))
            self.Y = np.eye(self.output_size)[target]
        else:
            self.problem = "regression"
            self.output_size = 1
            self.Y = target

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

    def transform(self, transform):
        trans = ["normalize", "standardize"]

        self.inputExistCheck()

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
            normalize = lambda x: (x - xmin) / (xmax - xmin)
            self.X = normalize(self.X)

        if transform == "standardize":
            xmean, xstd = np.mean(self.X), np.std(self.X)
            standardize = lambda x: (x - xmean) / xstd
            self.X = standardize(self.X)

        self.transformed = True
        self.transforms.append(transform)

        if self.verbose:
            print(f"Transform:\t{self.transforms[0].capitalize()[:-1]}ation")

    def split(self, test_split=1 / 7, shuffle=True, seed=None):
        self.inputExistCheck()

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False only")

        if shuffle:
            r = np.random.RandomState(seed=seed).permutation(self.samples)
            self.X, self.Y = self.X[r], self.Y[r]

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
            self.splitted = True
            del self.X, self.Y
        except:
            raise ValueError(f"{test_split} is not a valid test split. Default is 1/7")

        if self.verbose:
            print(
                f"Split:\t\t{int(self.X_train.shape[0]):,} ({1.0-test_split:.2%}) Train - {int(self.X_test.shape[0]):,} ({test_split:.2%}) Test"
            )
            print(f"Shuffled:\t{shuffle}")

    def addLayer(self, neurons=128, activation="relu", dropout=False):
        self.acts = ["relu", "tanh", "sigmoid", "softmax"]
        self.splitCheck()

        if neurons < 1 or neurons > 1024:
            raise ValueError(f"Number of neurons must be between 1 and 1024 not")

        if activation not in self.acts:
            raise ValueError(
                f"{activation} is not a valid activation function. Options: {self.acts}"
            )

        if not isinstance(dropout, bool):
            raise TypeError("dropout must be True or False only")

        self.hlayers += 1

        if self.hlayers == 1:
            self.n_input_size = self.input_size
            print(f"\n\tInput [{self.X_train.shape[1]}]")
        else:
            self.n_input_size = self.previous_layer_size

        self.params_count += self.n_input_size * neurons
        self.weights.append(0.10 * np.random.randn(self.n_input_size, neurons))
        self.biases.append(np.zeros(neurons))
        self.params_count += neurons
        self.previous_layer_size = neurons

        if self.verbose:
            if dropout:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout")
            else:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, activation="softmax"):

        if activation not in self.acts:
            raise ValueError(
                f"{activation} is not a valid activation function. Options: {self.acts}"
            )

        if self.hlayers < 1:
            raise NotImplementedError("No hidden layers detected. Try addLayer()")

        self.outputExists = True
        self.params_count += self.previous_layer_size * self.output_size
        self.output_weights = 0.10 * np.random.randn(
            self.previous_layer_size, self.output_size
        )
        self.output_biases = np.zeros(self.output_size)
        self.params_count += self.output_size

        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})\n")
            print(f"Parameters:\t{self.params_count:,}")

    def compile(
        self, valid_split=0.2, optimizer="adam", scorer="accuracy", early_stoppage=True
    ):
        self.outputExistCheck()
        opts = ["adam", "sgd", "rmsprop", "adadelta"]

    def train(self, batch_size=32, epochs=10):
        # 1 hidden layer only
        if self.hlayers == 1:
            self.last_output = np.dot(self.X_train, self.weights[0]) + self.biases[0]
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

        self.output = np.dot(self.last_output, self.output_weights) + self.output_biases
        print(self.output)

    def predict(self, transform):

        print(f"\nPredicting on {self.test_size:,} samples...")
        print("-------------------------------")

        if self.problem == "classification":
            self.output = np.argmax(self.output, axis=1)

    def evaluate(self):
        pass

    def splitCheck(self):
        if not self.splitted:
            raise NotImplementedError(
                f"You have not split the data into training and test sets yet. Use split()"
            )

    def inputExistCheck(self):
        if not self.inputExists:
            raise NotImplementedError("No input detected. Use input()")

    def outputExistCheck(self):
        if not self.outputExists:
            raise NotImplementedError(
                "Model has no output yet. Use output() to add output."
            )
