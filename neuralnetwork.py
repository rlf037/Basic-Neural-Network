import numpy as np
import dill
import tensorflow as tf

class NNet:

    def __init__(self, verbose=False):
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.verbose = verbose
        self.params = 0
        self.weights = []
        self.biases = []
        self.activations = []
        self.dropouts = []
        self.inputExists = False
        self.outputExists = False
        self.dropout_num = .2
        self.glorot = 6
        self.he = 2
        self.activators = ['relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu']
        self.optimizers = ["adam", "sgd", "rmsprop", "adadelta"]
        self.loss_functions = ['mae', 'mse', 'cce', 'scce', 'bce']
        self.scorers = ['accuracy', 'loss']
        if self.verbose:
            print("* Neural Network Initialised *\n")

    @staticmethod
    def Encode(labels):
        if str(labels.dtype)[:2] != '<U':
            raise TypeError(f"Cannot encode {labels.dtype} must be string/unicode format")
        output_size = len(np.unique(labels))
        count = 0
        output = []
        output_dict = {}
        total = set()
        code = {}
        for i in labels:
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
    def Normalize(data):
        xmax, xmin = np.amax(data), np.amin(data)
        normalize = lambda x: (x - xmin) / (xmax - xmin)
        return normalize(data)

    @staticmethod
    def Standardize(data):
        xmean, xstd = np.mean(data), np.std(data)
        standardize = lambda x: (x - xmean) / xstd
        return standardize(data)

    @staticmethod
    def Split(data, target, test_split=1/5, shuffle=True, seed=None):
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
            return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]
        else:
            raise ValueError(f"Test split {test_split} must be a value between (excluding) 0 and 1")

    def input(self, input_size):
        if not isinstance(input_size, tuple):
            raise TypeError("Input size must be a tuple")
        if not input_size:
            raise ValueError("Input size is None")
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
        self.sample_size = input_size[0]

        if self.verbose:
            if self.flatten:
                print(f"Flattened:\t({int(self.input_size**0.5)}, {int(self.input_size**0.5)}) -> {self.input_size}")

    def hidden(self, neurons=50, activation="relu", dropout=False):
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
        else:
            self.dropouts.append(dropout)

        # get amount of input neurons needed for this layer from the previous layer's output size
        if len(self.weights) == 0:
            # FIRST LAYER ONLY
            input_neurons = self.input_size
            if self.verbose:
                print("----------------------")
                print(f"\tInput [{self.input_size}]")
        else:
            # ALL OTHER LAYERS
            input_neurons = self.previous_output_size

        if activation == 'relu' or activation == 'leaky_relu':
            init = self.he
        else:
            init = self.glorot

        # parameter counter
        self.params += (input_neurons * neurons) + neurons
        # INITIALIZE WEIGHTS & BIASES
        self.weights.append(np.random.randn(input_neurons, neurons) * np.sqrt(init/input_neurons))
        self.biases.append(np.random.randn(1, neurons))
        # set this layer's output neurons for the next layer
        self.previous_output_size = neurons

        if self.verbose:
            if dropout:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout")
            else:
                print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, output_size, activation=None):
        if len(self.weights) == 0:
            raise NotImplementedError("No hidden layer(s) implemented yet - try hidden() first")
        if not isinstance(output_size, tuple):
            raise TypeError("Output size must be a tuple")
        if not output_size:
            raise ValueError("Output size is None")
        if output_size[0] != self.sample_size:
            raise ValueError(f"Input and Output have unequal sample sizes. ({self.sample_size} vs. {output_size[0]})")
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

        if activation == 'relu' or activation == 'leaky_relu':
            init = self.he
        else:
            init = self.glorot

        self.params += (self.previous_output_size * self.output_size) + self.output_size
        self.weights.append(np.random.randn(self.previous_output_size, self.output_size) * np.sqrt(init/self.previous_output_size))
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
            settings = locals()
            del settings['self']
            print(settings)

    @staticmethod
    def Activate(act, x, d=False):
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
            losses = []
            for pred, true in zip(y_pred, y_true):
                losses.append(np.sum((pred - true)**2) / y_true.size)
            return np.mean(losses)

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
        X_train, self.X_valid = (data[:self.train_size], data[-self.valid_size:])
        Y_train, self.Y_valid = (target[:self.train_size], target[-self.valid_size:])
        # reset batch size if greater than training size
        if self.batch_size >= self.train_size:
            self.batch_size = self.train_size
# ====================================================================================
        print(f'\nTraining on {self.train_size:,} samples...')

        # EPOCH ITERATION
        for i in range(self.epochs):
            
            # batch info
            sample_start = 0
            sample_end = self.batch_size
            current_batch_finished = False
            is_last_run = False

            # DROPOUT
            if not self.dropouts:
                pass
            else:
                self.dropout(self.dropout_num)

            # BATCH ITERATION
            while not current_batch_finished:
                #set the batch sample
                X, Y = X_train[sample_start:sample_end], Y_train[sample_start:sample_end]

                # get the output of a feed forward
                output = self.forward(X)

                # calculate loss from the output of the feed forward and the actual Y
                ####loss = self.Loss(self.loss, Y, output)
                loss = self.cce(Y, output)

                # ========== GRADIENT DESCENT [BACKPROPAGATION] ==========
                self.backprop(loss)

                print(f"Epoch {i+1}/{self.epochs}\tSamples {sample_end}/{self.train_size}\tTrain Loss {loss:4f}",
                end='\r')
                
                # setup next batch
                sample_start = sample_end
                # if the next batch will go equal or beyond the training sample size
                # set the end of the batch to the training size
                if sample_end + self.batch_size >= self.train_size:
                    sample_end = self.train_size
                    # if it was it's last run, it's now complete
                    if is_last_run:
                        current_batch_finished = True
                    # and set the batch loop to it's last run
                    is_last_run = True
                else:
                    # increase the end of the batch samples by a batch size
                    sample_end += self.batch_size

            # validate the newly optimized weights and biases with new data
            valid_loss, valid_acc = self.validate()

            #print loss and accuracy after each epoch with valid data
            print(f"\nVal Loss: {loss:.4f} | Val Accuracy: {valid_acc:.3%}")
            # self.output = np.concatenate(output_list)

    def dropout(self, num):
        for index, drop in enumerate(self.dropouts):
            if drop:
                drops = np.random.binomial(1, 1-num, self.weights[index].shape)
                self.weights[index] *= drops

    def forward(self, X):
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            X = np.dot(X, weight) + bias
            X = self.Activate(activation, X)
        return X

    def backprop(self, loss):
        momentum = 0.9

        #error_grad = 
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
