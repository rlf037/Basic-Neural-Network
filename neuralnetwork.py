import numpy as np
import dill, time

class NNet:

    def __init__(self, verbose=False):
        """
        Class Methods:-
        input:
        hidden:
        output:
        compile:
        evaluate:
        predict:
        save:
        forward:
        dropout:

        Static Methods:-
        Normalize:
        Split:
        Encode:
        Standardize:
        Load:
        Activate:
        Loss:
        """
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")
        self.verbose = verbose
        self.params = 0
        self.weights = []
        self.biases = []
        self.activations = []
        self.dropouts = []
        self.inputExists = False
        self.outputExists = False
        self.glorot = 6
        self.he = 2
        self.valid_loss = .0
        self.valid_acc = .0
        self.activators = ['relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu']
        self.optimizers = ["adam", "sgd", "rmsprop", "adadelta"]
        self.loss_functions = ['mae', 'mse', 'categorical_crossentropy', 'scce', 'bce']
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
            if i in output_dict: output.append(output_dict[i])
            else:
                output_dict[i] = count
                output.append(output_dict[i])
                count += 1
                code[len(total)] = i
                total.add(i)
        #return np.eye(output_size)[output], code
        return output, code

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
        if not isinstance(input_size, tuple): raise TypeError("Input size must be a tuple")
        if not input_size: raise ValueError("Input size is None")
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
                print(f"Flattened: ({int(self.input_size**0.5)}, {int(self.input_size**0.5)}) -> {self.input_size}")

    def hidden(self, neurons=50, activation="relu", dropout=False):
        if not self.inputExists: raise NotImplementedError("No input layer implemented yet - try input() first")
        if neurons<1 or neurons>1024 or not isinstance(neurons, int): raise ValueError(f"{neurons} must be an integer between 1 and 1024")
        if activation.lower() not in self.activators: raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")
        else: self.activations.append(activation.lower())
        if dropout>0 and dropout<1 or not dropout: self.dropouts.append(dropout)
        else: raise ValueError(f" Dropout ({dropout}) must be a value between 0 and 1 (excluding)")

        # get amount of input neurons needed for this layer from the previous layer's output size
        if len(self.weights) == 0:
            # FIRST LAYER ONLY
            input_neurons = self.input_size
            if self.verbose:
                print("----------------------")
                print(f"\tInput [{self.input_size}]")
        else: input_neurons = self.previous_output_size
            
        # parameter counter
        self.params += (input_neurons * neurons) + neurons
        # INITIALIZE WEIGHTS & BIASES
        np.random.seed(0)
        random.seed(0)
        self.weights.append(0.01 * np.random.randn(input_neurons, neurons))
        self.biases.append(np.zeros((1, neurons)))
        # set this layer's output neurons for the next layer
        self.previous_output_size = neurons

        if self.verbose:
            if dropout: print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout [{dropout}]")
            else: print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, output_size, activation=None):
        if len(self.weights) == 0: raise NotImplementedError("No hidden layer(s) implemented yet - try hidden() first")
        if not isinstance(output_size, tuple): raise TypeError("Output size must be a tuple")
        if not output_size: raise ValueError("Output size is None")
        if output_size[0] != self.sample_size: raise ValueError(f"Input and Output have unequal sample sizes. ({self.sample_size} vs. {output_size[0]})")
        if activation.lower() not in self.activators: raise ValueError(f"{activation} is not a valid activation function. Options: {self.activators}")
        else: self.activations.append(activation.lower())

        # set output size to the shape[1] of the target data
        try:    self.output_size = output_size[1]
        except: self.output_size = 1

        self.outputExists = True
        
        np.random.seed(0)
        random.seed(0)
        self.params += (self.previous_output_size * self.output_size) + self.output_size
        self.weights.append(0.01 * np.random.randn(self.previous_output_size, self.output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        
        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})")
            print("----------------------")
            print(f"Total Parameters: {self.params:,}")

    def compile(self, optimizer="adam", loss=None, learn_rate=1e-3):
        if not self.outputExists: raise NotImplementedError("No output layer implemented yet - try output() first")
        if optimizer.lower() not in self.optimizers: raise ValueError(f"{optimizer} is not a valid optimizer. Options: {self.optimizers}")
        else: self.optimizer = optimizer.lower()
        if loss.lower() not in self.loss_functions: raise ValueError(f"{loss} is not a valid loss function. Options: {self.loss_functions}")
        else: self.loss = loss.lower()
        self.learn_rate = learn_rate
        if self.verbose:
            s = locals()
            print(f"Optimizer: {s['optimizer']}\nLoss: {s['loss']}\nLearning Rate: {s['learn_rate']}")
            del s

    def train(self, data, target, batch_size=64, epochs=15, valid_split=None):
        if data is None: raise ValueError("X is None")
        if target is None: raise ValueError("Y is None")
        if not isinstance(data, np.ndarray):
            try: data = np.array(data)
            except: raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if not isinstance(target, np.ndarray):
            try: target = np.array(target)
            except: raise TypeError(f"Cannot convert Y of type {type(target)} to a NumPy array")
        if batch_size>0 and isinstance(batch_size, int): self.batch_size = batch_size
        else: raise ValueError(f"Batch size {batch_size} must be an integer greater than 1")
        if epochs > 0 and epochs < 1000 and isinstance(epochs, int): self.epochs = epochs
        else: raise ValueError(f"({epochs}) Number of epochs must be an integer between 1 and 999")
        if valid_split:
            if not (valid_split>0) or not (valid_split <1): 
                raise ValueError(f"({valid_split}) Validation split must be a value between 0 and 1")
        self.valid_split = valid_split
        if data.shape[0] != target.shape[0]: raise ValueError(f"X and Y do not have equal samples. ({data.shape[0]} vs. {target.shape[0]})")
        # flatten data if applicable before being trained
        if self.flatten: data = data.reshape(data.shape[0], -1)
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
        # ====================================================================================
        if self.verbose:
            s = locals()
            print(f"Batch Size: {s['batch_size']}\nEpochs: {s['epochs']}\nValidation Split: {s['valid_split']}")
            del s

        print(f'\nTrain on {self.train_size} samples, validate on {self.valid_size} samples:')

        progress_bar = 23
        # EPOCH ITERATION
        for i in range(self.epochs):
            start_etime = time.time()

            print(f"\nEpoch {i+1}/{self.epochs}")

            # shuffle the data each epoch
            r_perm = np.random.RandomState().permutation(self.train_size)
            X_train, Y_train = X_train[r_perm], Y_train[r_perm]

            # batch info
            start = 0
            end = self.batch_size
            current_batch_finished = False
            is_last_run = False
            batches = int(np.ceil(self.train_size/self.batch_size))
            count = 1
            if self.valid_split: self.valid_loss = self.valid_acc = 0

            # DROPOUT
            if len(self.dropouts)>0:
                self.dropout()

            # BATCH ITERATION
            while not current_batch_finished:

                # set the batch sample
                X, Y = X_train[start:end], Y_train[start:end]

                # feed forward the batch
                output = self.forward(X)

                # calculate loss from predictons of the feed forward and the actual Y
                loss = self.Loss(output, Y)

                # accuracy of the batch
                preds = np.argmax(output, axis=1)
                acc = np.mean(preds==Y)

                # get the delta weights and biases (backpropagation)
                delta_weights, delta_biases = self.backward(output, Y)

                # update weights and biases (gradient descent)
                for i, (dw, db) in enumerate(zip(delta_weights, delta_biases)):
                    self.weights[i] -= self.learn_rate * dw
                    self.biases[i] -= self.learn_rate * db

                if not current_batch_finished:
                    end_etime = start_etime

                # progress bar
                equals = ((count-1)/(batches-0))*(progress_bar-0)+0
                if int(equals) == progress_bar-1: equals = '='*int(equals)
                else: equals = '='*int(equals)+'>'

                print(f"{end}/{self.train_size} [{equals}] - {int(end_etime-start_etime)}s - loss: {np.mean(loss):.4f} - accuracy: {acc:.3%} - val_loss: {np.mean(self.valid_loss):.4f} - val_accuracy: {self.valid_acc:.3%}", end='\r')
               
                # setup next batch
                start = end

                """if the next batch will go equal or beyond the total training 
                   size set the end of the batch to the training size"""
                if end + self.batch_size >= self.train_size:
                    end = self.train_size
                    # if it was it's last run, it's now complete
                    if is_last_run: current_batch_finished = True
                    # set the batch loop to it's last run
                    is_last_run = True
                # increase the end of the batch samples by a batch size
                else: end += self.batch_size
                count += 1
                # ==== END BATCH ITERATION =====
 
            # validate the newly optimized weights and biases with new data
            if self.valid_split: self.valid_loss, self.valid_acc = self.validate()

            end_etime = time.time()
            print(f"{end}/{self.train_size} [{equals}] - {int(end_etime-start_etime)}s - loss: {np.mean(loss):.4f} - accuracy: {acc:.3%} - val_loss: {np.mean(self.valid_loss):.4f} - val_accuracy: {self.valid_acc:.3%}", end='\r')
            # === END EPOCH ITERATION ==== 

    def Activate(self, act, x, d=False, layer=None):
        self.act_inputs.append(x)
        if act == 'sigmoid':
            if d: 
                return sig * (1. - sig)
            else: return 1./(1. + np.exp(-x))
        if act == 'tanh':
            if d: return 1.0 - x**2
            else: return np.tanh(x)

        if act == 'relu':
            if not d:
                return np.maximum(0, x)
            else:
                x[self.act_inputs[layer]<=0] = 0
                return x

        if act == 'leaky_relu':
            if not d:
                return np.maximum(x*.01, x)

            else:
                x[x < 0] = .01
                x[x >=0] = 1
                return x

        if act == 'softmax':
            if not d:
                exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_values / np.sum(exp_values, axis=1, keepdims=True)

            else:
                return x

    def Loss(self, y_pred, y_true, d=False):
        if self.loss == 'categorical_crossentropy':
            # Number of samples in a batch
            samples = y_pred.shape[0]
            if not d:
                # Probabilities for target values - only if categorical labels
                if len(y_true.shape) == 1:
                    y_pred = y_pred[range(samples), y_true]
                # Losses
                negative_log_likelihoods = -np.log(y_pred)
                # Mask values - only for one-hot encoded labels
                if len(y_true.shape) == 2:
                    negative_log_likelihoods *= y_true
                # Overall loss
                data_loss = np.sum(negative_log_likelihoods) / samples
                return data_loss
            else:
                y_pred[range(samples), y_true] -= 1
                y_pred /= samples
                return y_pred
        if self.loss == 'mae': return np.sum(np.absolute(y_pred - y_true))
        if self.loss == 'mse': #L2
            losses = []
            for pred, true in zip(y_pred, y_true):
                losses.append(np.sum((pred - true)**2))
            return losses

    def dropout(self):
        for i, drop in enumerate(self.dropouts):
            if drop:
                drops = np.random.binomial(1, 1-drop, self.weights[i].shape)
                self.weights[i] *= drops

    def forward(self, X):
        self.inputs = []
        self.act_inputs = []
        for i, (w, b, a) in enumerate(zip(self.weights, self.biases, self.activations)):
            self.inputs.append(X)
            X = np.dot(X, w) + b
            X = self.Activate(a, X)
        return X

    def backward(self, X, Y):
        # initialise delta weights/biases arrays
        delta_weights = [np.ones(w.shape) for w in self.weights]
        delta_biases = [np.ones(b.shape) for b in self.biases]
        
        X = self.Loss(X, Y, d=True)

        # backwards pass calculating delta losses
        for i, a in reversed(list(enumerate(self.activations))):
            
            X = self.Activate(a, X, d=True, layer=i)
            
            # Gradients on parameters
            delta_weights[i] = np.dot(self.inputs[i].T, X)
            delta_biases[i] = np.sum(X, axis=0, keepdims=True)
            # Gradient on values
            X = np.dot(X, self.weights[i].T)
        
        return delta_weights, delta_biases

    def validate(self):
        predictions = self.forward(self.X_valid)
        valid_loss = self.Loss(predictions, self.Y_valid)
        if self.output_size > 1:
            predictions = np.argmax(predictions, axis=1)
            #targets = np.argmax(self.Y_valid, axis=1)
            targets = self.Y_valid
        else: targets = self.Y_valid
        valid_acc = np.mean(predictions == targets)
        return valid_loss, valid_acc

    # Automatic predictions and evaluations based on test data
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
        # flatten data if applicable before being trained
        if self.flatten: data = data.reshape(data.shape[0], -1)

        print(f"\n\nTesting Set Evaluation:")
        print("-----------------------")

        predictions = self.forward(data)
        loss = self.Loss(predictions, target)

        if self.output_size > 1:
            predictions = np.argmax(predictions, axis=1)
            #targets = np.argmax(target, axis=1)
            targets = target
        else: targets = target

        self.test_acc = np.mean(predictions == targets)
        print(f"Test Acc:\t{np.sum(predictions == targets)}/{target.shape[0]} ({self.test_acc:.2%})")
        print(f"Test Loss:\t{np.mean(loss):.4f}\n")

    def predict(self, data):
        if data is None: raise ValueError("X is None")
        if not isinstance(data, np.ndarray):
            try: data = np.array(data)
            except: raise TypeError(f"Cannot convert X of type {type(data)} to a NumPy array")
        if self.flatten:
            if len(data.shape) == 2: data = data.flatten()
            else: data = data.reshape(data.shape[0], -1)

        output = self.forward(data)

        if self.output_size > 1:
            if output.shape[0] > 1: prediction = np.argmax(output, axis=1)
            else: prediction = np.argmax(output)
            score = np.amax(output)
            return prediction, score
        else: return prediction

    def save(self, file):
        path = 'models/' + file + '.pkl'
        with open(path, 'wb') as f: dill.dump(self, f)
        print(f"'{file.upper()}' model saved.")

    @staticmethod
    def Load(file):
        path = 'models/' + file + '.pkl'
        with open(path, 'rb') as f:
            print(f"'{file.upper()}' model loaded.\n")
            return dill.load(f)