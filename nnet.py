import numpy as np

class NeuralNetwork:
    def __init__(self, verbose=False, process=None):

        ## PRE-PROCESSING
        self.count = 0
        self.outputs = []
        self.seen = {}
        self.code = {}
        self.one_hot = None
        self.total = set()

        if not isinstance(verbose, bool): raise TypeError("verbose is T/F")

        # NEURAL NET
        self.verbose = verbose
        self.process = process
        self.params = 0
        self.weights = []
        self.biases = []
        self.activations = []
        self.dropouts = []
        self.train_hist_loss = []
        self.train_hist_acc = []
        self.val_hist_loss = []
        self.val_hist_acc = []
        self.activators = {'relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu'}
        self.optimizers = {'adam', 'sgd', 'rmsprop', 'adadelta'}
        self.losses = {'mae', 'mse', 'cce', 'scce', 'bce', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy'}
        self.inputExists = False
        self.outputExists = False
        self.valid_loss = .0
        self.valid_acc = .0
        
        if self.verbose:
            print(f'* [{self.process.capitalize()}] Neural Network Initialised *')

    def input(self, input_size=None):

        if not isinstance(input_size, int): raise TypeError('input_size must be an int')

        self.input_size = input_size
        self.inputExists = True
        
        if self.verbose:
                print('-------------------------------')
                print(f'\tInput [{self.input_size}]')

    def hidden(self, neurons=50, activation="relu", dropout=False):

        if not self.inputExists: raise NotImplementedError("No input layer implemented yet - try input() first")
        if neurons<1 or neurons>1024 or not isinstance(neurons, int): raise ValueError(f"neurons ({neurons}) must be an integer between 1 and 1024")
        if activation.lower() not in self.activators: raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")
        if dropout and (dropout<=0 or dropout>=1): raise ValueError(f"Dropout ({dropout}) must be a value between (excluding) 0 and 1")
        
        # get amount of input neurons needed for this layer from the previous layer's output size
        if len(self.weights) == 0:
            # if it's the first hidden layer - use input size
            input_neurons = self.input_size
        else: input_neurons = self.previous_output_size

        self.params += (input_neurons * neurons) + neurons

        # INITIALIZE WEIGHTS & BIASES
        self.weights.append(np.random.randn(input_neurons, neurons) * np.sqrt(2/input_neurons))
        self.biases.append(np.zeros((1, neurons)))
        self.activations.append(activation.lower())
        self.dropouts.append(dropout)

        # set this layer's output neurons for the next layer
        self.previous_output_size = neurons

        if self.verbose:
            if dropout: print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout {dropout:.0%}")
            else: print(f"\t\t|\t\t\n\tHidden [{neurons}] ({activation})")

    def output(self, output_size=None, activation=None):
        if len(self.weights) == 0: raise NotImplementedError("No hidden layer(s) implemented yet - try hidden() first")
        if not isinstance(output_size, int): raise TypeError("output_size must be a int")
        if activation.lower() not in self.activators: raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")

        self.outputExists = True

        self.output_size = output_size
        self.params += (self.previous_output_size * self.output_size) + self.output_size
        self.weights.append(np.random.randn(self.previous_output_size, self.output_size) * np.sqrt(2/self.previous_output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        self.activations.append(activation.lower())
        
        if self.verbose:
            print(f"\t\t|\t\t\n\tOutput [{self.output_size}] ({activation})")
            print("-------------------------------")
            print(f"Total Parameters: {self.params:,}")

    def compile(self, optimizer="adam", loss=None, learn_rate=1e-1):

        if not self.outputExists: raise NotImplementedError("No output layer implemented yet - try output() first")
        if optimizer.lower() not in self.optimizers: raise ValueError(f"{optimizer} is not a valid optimizer. Options: {self.optimizers}")
        if loss.lower() not in self.losses: raise ValueError(f"{loss} is not a valid loss function. Options: {self.losses}")

        self.loss = loss.lower()
        self.learn_rate = learn_rate
        self.optimizer = optimizer.lower()

        if (self.one_hot and (self.loss != 'sparse_categorical_crossentropy')):
            raise ValueError('SCCE is needed for one-hot encoding.')

        if self.verbose:
            s = locals()
            del s['self']
            print(s)

    def train(self, data, target, batch_size=64, epochs=15, valid_split=0.1, early_stopping=None, save_weights=True):

        if data is None: raise ValueError("X is None")
        if target is None: raise ValueError("Y is None")
        if not isinstance(data, np.ndarray):
            try: data = np.array(data)
            except: raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try: target = np.array(target)
            except: raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")

        if not isinstance(batch_size, int) or batch_size<=0: raise ValueError(f"batch_size {batch_size} must be an integer greater than 0")
        if epochs<=0 or epochs>=1000 or not isinstance(epochs, int): ValueError(f"Number of epochs ({epochs}) must be an integer between (including) 1 and 999")
        if valid_split:
            if not (valid_split>0) or not (valid_split <1): 
                raise ValueError(f"({valid_split}) Validation split must be a value between (excluding) 0 and 1")

        if data.shape[0] != target.shape[0]: raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")

        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_split = valid_split

        self.best_weights = None
        self.best_biases = None

        if self.verbose:
            s = locals()
            del s['self']
            del s['data']
            del s['target']
            print(s)

         # split into training and validation data
        if valid_split:
            self.valid_size = int(data.shape[0] * self.valid_split)
            self.train_size = data.shape[0] - self.valid_size
            X_train, self.X_valid = (data[:self.train_size], data[-self.valid_size:])
            Y_train, self.Y_valid = (target[:self.train_size], target[-self.valid_size:])
        else:
            self.train_size = data.shape[0]
            self.valid_size = 0
            X_train = data
            Y_train = target

        # reset batch size if greater than training size
        if self.batch_size >= self.train_size: self.batch_size = self.train_size
        progress_bar = 30
        total_batches = int(np.ceil(self.train_size/self.batch_size))

        best_acc = 0
        no_improve = 0

        print(f'\nTrain on {self.train_size} samples, validate on {self.valid_size} samples:')
        # ====================================================================================

        # EPOCH ITERATION
        for epoch in range(1, self.epochs+1):

            # Epoch 1/15 etc.
            print(f"Epoch {epoch}/{self.epochs}")

            # shuffle the data each epoch, except the first epoch
            if epoch != 1:
                rp = np.random.RandomState().permutation(self.train_size)
                X_train, Y_train = X_train[rp], Y_train[rp]

            # batch info
            start = 0
            end = self.batch_size
            current_batch_finished = False
            is_last_run = False
            current_batch = 1

            loss_list = []
            acc_list = []

            # DROPOUT
            if len(self.dropouts)>0:
                self.dropout()

            # BATCH ITERATION
            while not current_batch_finished:

                # set the batch sample
                X, Y = X_train[start:end], Y_train[start:end]

                # feed forward the batch
                output = self.forward(X, train=True)

                # calculate loss of the batch
                loss = self.Loss(output, Y)
                loss_list.append(loss)

                # accuracy of the batch
                preds = np.argmax(output, axis=1)
                # ONE HOT CODED
                if isinstance(Y[0], np.ndarray):
                    targets = np.argmax(Y, axis=1)
                else:
                    targets = Y
                acc = np.mean(preds == targets)
                acc_list.append(acc)

                # get the delta weights and biases (backpropagation)
                delta_weights, delta_biases = self.backward(output, Y)

                # update weights and biases (gradient descent)
                for i, (dw, db) in enumerate(zip(delta_weights, delta_biases)):
                    self.weights[i] -= self.learn_rate * dw
                    self.biases[i] -= self.learn_rate * db

                if total_batches == 1:
                    pb = '=' * progress_bar
                else:
                    # mapped current_batch -> batches to 1 -> progress_bar
                    pb = int(((current_batch - 1) / (total_batches -1 )) * (progress_bar -1 ) + 1)
                    if pb == progress_bar:
                        pb = '=' * pb
                    else:
                        pb = '='* pb + '>' + '.' * (progress_bar - pb - 1)

                print(f"{end}/{self.train_size} [{pb}] - loss: {loss:.5f} - accuracy: {acc:.5f}", end='\r')

                # setup next batch
                start = end
                """if the next batch will go equal or beyond the total training
                   size set the end of the batch to the training size"""
                if end + self.batch_size >= self.train_size:
                    end = self.train_size
                    # if it was it's last run, it's now complete
                    if is_last_run:
                        current_batch_finished = True
                    # set the batch loop to it's last run
                    is_last_run = True
                # increase the end of the batch samples by a batch size
                else:
                    end += self.batch_size
                current_batch += 1
                # ==== END BATCH ITERATION =====

            # validate the newly optimized weights and biases with new data
            if self.valid_split:
                self.valid_loss, self.valid_acc = self.validate()
            # add the current loss/acc to history to plot later on
            loss = np.mean(loss_list)
            acc = np.mean(acc_list)
            self.train_hist_loss.append(loss)
            self.train_hist_acc.append(acc)
            self.val_hist_loss.append(self.valid_loss)
            self.val_hist_acc.append(self.valid_acc)
            # print the validation loss & acc too
            print(f"{end}/{self.train_size} [{pb}] - loss: {loss:.5f} - accuracy: {acc:.5f} - val_loss: {self.valid_loss:.5f} - val_accuracy: {self.valid_acc:.5f}")

            # === EARLY STOPPING & SAVE BEST MODEL ===
            if (self.valid_acc <= best_acc):
                no_improve += 1
            else:
                best_acc = self.valid_acc
                if save_weights:
                    self.best_weights = self.weights
                    self.best_biases = self.biases
                no_improve = 0

            if (no_improve == early_stopping):
                print(f'\n*Early Stoppage* Model has not improved in {early_stopping} epochs.')
                break
            # === END EPOCH ITERATION ====

        if (self.valid_acc < best_acc):
            self.weights = self.best_weights
            self.biases = self.best_biases
            print(f'\nLoaded the weights that achieved {best_acc:.5f} accuracy.')

    def Activate(self, act, x, dx=False, layer=None):

        if act == 'relu':
            if not dx:
                return np.maximum(0, x)
            else:
                x[self.act_inputs[layer] <= 0] = 0
                return x

        if act == 'leaky_relu':
            if not dx:
                return np.maximum(.01, x)
            else:
                x[self.act_inputs[layer] < 0] = .01
                return x

        if act == 'tanh':
            if not dx:
                return np.tanh(x)
            else:
                return 1. - self.act_inputs[layer]**2

        if act == 'sigmoid':
            sig = 1. / (1. + np.exp(-x))
            if not dx:
                return sig
            else:
                return sig * (1. - sig)

        if act == 'softmax':
            if not dx:
                exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_values / np.sum(exp_values, axis=1, keepdims=True)
            else:
                return x
            
        if act == 'linear':
            return x

    def Loss(self, x, y, dx=False):

        samples = x.shape[0]

        if self.loss == 'mean_absolute_error' or self.loss == 'mae':
            if not dx:
                return np.mean(np.absolute(x - y))
            else:
                raise NotImplementedError()

        if self.loss == 'mean_squared_error' or self.loss == 'mse':
            if not dx:
                return np.mean((x - y)**2)
            else:
                raise NotImplementedError()

        if self.loss == 'binary_crossentropy' or self.loss == 'bce':
            if not dx:
                x = x[range(samples), y]
                return np.mean(-np.log(x))
            else:
                x[range(samples), y] -= 1
                return x / samples

        if self.loss == 'categorical_crossentropy' or self.loss == 'cce':
            if not dx:
                x = x[range(samples), y]
                return np.mean(-np.log(x))
            else:
                x[range(samples), y] -= 1
                return x / samples

        if self.loss == 'sparse_categorical_crossentropy' or self.loss == 'scce':
            y = np.argmax(y, axis=1)
            if not dx:
                x = x[range(samples), y]
                return np.mean(-np.log(x) * (y))
            else:
                x[range(samples), y] -= 1
                return x / samples

    def forward(self, X, train=False):

        if train:
            self.inputs = []
            self.act_inputs = []

        for w, b, a in zip(self.weights, self.biases, self.activations):
            if train:
                self.inputs.append(X)
            X = X @ w + b
            if train:
                self.act_inputs.append(X)
            X = self.Activate(a, X)
        return X

    def backward(self, X, Y):

        delta_weights = []
        delta_biases = []

        # calculate derivitive loss
        X = self.Loss(X, Y, dx=True)

        # backwards pass calculating derivitive values at each layer
        for layer, _ in reversed(list(enumerate(self.activations))):

            # derivitive loss of the activation functions
            X = self.Activate(_, X, dx=True, layer=layer)

            # delta values calculated here
            delta_weights.insert(0, self.inputs[layer].T @ X)
            delta_biases.insert(0, np.sum(X, axis=0, keepdims=True))

            # set X for the next layer
            X = X @ self.weights[layer].T

        return delta_weights, delta_biases

    def dropout(self):
        for layer, drop in enumerate(self.dropouts):
            if drop:
                self.weights[layer] *= np.random.binomial(1, 1 - drop, self.weights[layer].shape) / (1-drop)

    def validate(self):

        predictions = self.forward(self.X_valid)
        valid_loss = self.Loss(predictions, self.Y_valid)

        # ONE HOT CODED TARGETS
        if self.one_hot:
            targets = np.argmax(self.Y_valid, axis=1)
        else:
            targets = self.Y_valid

        predictions = np.argmax(predictions, axis=1)
            
        valid_acc = np.mean(predictions == targets)
        return valid_loss, valid_acc

    def evaluate(self, data, target):

        if data is None: raise ValueError("X is None")
        if target is None: raise ValueError("Y is None")

        if not isinstance(data, np.ndarray):
            try: data = np.array(data)
            except: raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try: target = np.array(target)
            except: raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")

        if data.shape[0] != target.shape[0]: raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")

        print(f"\nTesting Set Evaluation:")
        print("-----------------------")

        predictions = self.forward(data)
        loss = self.Loss(predictions, target)

        predictions = np.argmax(predictions, axis=1)

        if self.one_hot:
            target = np.argmax(target, axis=1)

        self.test_acc = np.mean(predictions == target)
        print(f"Test Acc:\t{np.sum(predictions == target)}/{target.shape[0]} ({self.test_acc:.2%})")
        print(f"Test Loss:\t{loss:.4f}\n")

    def predict(self, data):

        if data is None: raise ValueError("X is None")

        if not isinstance(data, np.ndarray):
            try: data = np.array(data)
            except: raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")

        prediction = self.forward(data)
        return np.argmax(prediction), np.amax(prediction)

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            plt.plot(self.train_hist_loss)
            plt.plot(self.val_hist_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model Loss')
            plt.legend(['Training', 'Validation'], loc='best')
            plt.show()

            plt.plot(self.train_hist_acc)
            plt.plot(self.val_hist_acc)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy')
            plt.legend(['Training', 'Validation'], loc='best')
            plt.show()
        except:
            print("Error: 'matplotlib' is required to plot.")

    def save(self, file):

        path = 'models/' + file

        np.savez_compressed(path, 
            weights=np.array(self.weights, dtype=list),
            input_biases=self.biases[:-1],
            output_biases=self.biases[-1],
            activations=self.activations,
            one_hot=self.one_hot,
            code=self.code)

        print(f"'{file.upper()}' model saved.")

    ### PRE-PROCESSING FUNCTIONS

    def encode(self, labels, one_hot=False):

        self.one_hot = one_hot

        for label in labels:

            if label in self.seen:
                self.outputs.append(self.seen[label])
            else:
                self.seen[label] = self.count
                self.outputs.append(self.seen[label])
                self.count += 1
                self.code[len(self.total)] = label
                self.total.add(label)

        if one_hot:
            return np.eye(len(self.total))[self.outputs]
        else:
            return np.array(self.outputs)

    def decode(self, data):

        try:
            iter(data)
            if self.one_hot:
                pos = np.argmax(data)
            return self.code[pos]

            return [self.code[x] for x in data]
        except:
            return self.code[data]

    def distribution(self, data):

        if self.one_hot:
            data = np.argmax(data, axis=1)

        freq = {}
        for x in data:
            if x in freq.keys():
                freq[x] += 1
            else:
                freq[x] = 1

        for k, v in freq.items():
            freq[k] = round((v/len(data))*100, 3)
        return freq

    @staticmethod
    def Normalize(data):

        xmin, xmax = np.amin(data), np.amax(data)
        minmax = lambda x: (x - xmin) / (xmax - xmin)

        return minmax(data)

    @staticmethod
    def Standardize(data):

        xmean, xstd = np.mean(data), np.std(data)
        standardize = lambda x: (x - xmean) / xstd

        return standardize(data)

    def split(self, data, target, test_split=1/7, shuffle=True, seed=None, stratify=True):

        if data is None: raise ValueError("X is None")
        if target is None: raise ValueError("Y is None")
        if not isinstance(shuffle, bool): raise TypeError("shuffle is T/F")

        if stratify and not shuffle:
            raise ValueError("'shuffle' must be true if using 'stratify'.")

        if data.shape[0] != target.shape[0]: raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")

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

        if test_split<=0 or test_split >=1: raise ValueError(f"test_split ({test_split}) must be a value between (excluding) 0 and 1")

        samples = data.shape[0]

        test_size = int(samples * test_split)
        train_size = samples - test_size

        if stratify:

            import math
            from itertools import cycle
            spinner = cycle(['-', '\\', '|', '/'])
            tol = 0.3

            print("Stratifying the train/test split...")
        
            dist = self.distribution(target)

            while True:
                
                failed = False
                
                print('\b' + next(spinner), end='')

                perm = np.random.RandomState(seed=seed).permutation(samples)
                data, target = data[perm], target[perm]
                target_train, target_test = target[:train_size], target[-test_size:]

                dist2 = self.distribution(target_train)
                dist3 = self.distribution(target_test)

                for i in range(10):
                    if math.isclose(dist[i], dist2[i], abs_tol=tol) and math.isclose(dist[i], dist3[i], abs_tol=tol):
                        continue
                    else:
                        failed = True;

                if not failed:
                    print('\b', end='')
                    return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]

        else:
            if shuffle:
                perm = np.random.RandomState(seed=seed).permutation(samples)
                data, target = data[perm], target[perm]
            return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]

class LoadModel(NeuralNetwork):

    def __init__(self, file):

        path = 'models/' + file + '.npz'
        self.biases = []
        with np.load(path, allow_pickle=True) as data:
            self.weights = data['weights']
            self.biases.append(data['input_biases'][0])
            self.biases.append(data['output_biases'])
            self.activations = data['activations']
            self.one_hot = data['one_hot']
            self.code = data['code'][()]

        print(f"'{file.upper()}' model loaded.")

