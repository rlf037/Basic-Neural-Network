import numpy as np
import dill, time

class NN:
    def __init__(self, verbose=False):

        self.verbose = verbose
        self.params = 0
        self.weights = []
        self.biases = []
        self.activations = []
        self.dropouts = []
        self.valid_loss = .0
        self.valid_acc = .0
        self.activators = ['relu', 'tanh', 'sigmoid', 'softmax', 'leaky_relu']
        self.optimizers = ["adam"]
        self.losses = ['mae', 'mse', 'cce', 'scce', 'bce', 'categorical_crossentropy', 'sparse_categorical_crossentrophy', 'binary_crossentrophy']
        self.progress_bar = 30
        if self.verbose:
            print("* Neural Network Initialised *")

    def input(self, input_size=None):

        self.input_size = input_size

        if self.verbose:
                print("---------------------------")
                print(f"\tInput [{self.input_size}]")

    def hidden(self, neurons=50, activation="relu", dropout=False):
        
        if len(self.weights) == 0: input_neurons = self.input_size
        else: input_neurons = self.previous_output_size
        self.params += (input_neurons * neurons) + neurons
        self.weights.append(np.random.randn(input_neurons, neurons) * np.sqrt(2/input_neurons))
        self.biases.append(np.zeros((1, neurons)))
        self.activations.append(activation.lower())
        self.dropouts.append(dropout)
        self.previous_output_size = neurons

        if self.verbose:
            if dropout: print(f"\t\t|\t\t\nHidden [{neurons}] ({activation}) - Dropout [{dropout:.0%}]")
            else: print(f"\t\t|\t\t\nHidden [{neurons}] ({activation})")

    def output(self, output_size=None, activation=None):

        self.output_size = output_size
        self.params += (self.previous_output_size * self.output_size) + self.output_size
        self.weights.append(np.random.randn(self.previous_output_size, self.output_size) * np.sqrt(2/self.previous_output_size))
        self.biases.append(np.zeros((1, self.output_size)))
        self.activations.append(activation.lower())
        
        if self.verbose:
            print(f"\t\t|\t\t\nOutput [{self.output_size}] ({activation})")
            print("----------------------------")
            print(f"Total Parameters: {self.params:,}")

    def compile(self, optimizer="adam", loss=None, learn_rate=1e-3):

        self.loss = loss.lower()
        self.learn_rate = learn_rate

        if self.verbose:
            s = locals()
            print(f"Optimizer: {s['optimizer'].upper()} & Learning Rate: {s['learn_rate']}")

    def train(self, data, target, batch_size=64, epochs=15, valid_split=0.1):

        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_split = valid_split

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

        if self.verbose:
            s = locals()
            print(f"Batch Size: {s['batch_size']} - Epochs: {s['epochs']} - Validation Split: {s['valid_split']}")
            del s
        # ====================================================================================

        print(f'\nTrain on {self.train_size} samples, validate on {self.valid_size} samples:')

        # EPOCH ITERATION
        for i in range(self.epochs):

            # time each epoch
            start_etime = time.time()

            print(f"\nEpoch {i+1}/{self.epochs}")

            # shuffle the data each epoch, except the first epoch
            if i != 0:
                r_perm = np.random.RandomState().permutation(self.train_size)
                X_train, Y_train = X_train[r_perm], Y_train[r_perm]

            # batch info
            start = 0
            end = self.batch_size
            current_batch_finished = False
            is_last_run = False
            batches = int(np.ceil(self.train_size/self.batch_size))
            count = 1

            # DROPOUT
            if len(self.dropouts)>0:
                self.dropout()

            # BATCH ITERATION
            while not current_batch_finished:

                # set the batch sample
                X, Y = X_train[start:end], Y_train[start:end]

                # feed forward the batch
                output = self.forward(X)

                # calculate loss
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
                equals = ((count-1)/(batches-0))*(self.progress_bar-0)+0
                if int(equals) == self.progress_bar-1: equals = '='*int(equals)
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

    def Activate(self, act, x, der=False, layer=None):
        self.act_inputs.append(x)
        if act == 'sigmoid':
            if der: 
                return sig * (1. - sig)
            else: return 1./(1. + np.exp(-x))

        if act == 'tanh':
            if not der: return np.tanh(x)
            else: return 1.0 - x**2

        if act == 'relu':
            if not der:
                return np.maximum(0, x)
            else:
                x[self.act_inputs[layer]<=0] = 0
                return x

        if act == 'leaky_relu':
            if not der:
                return np.maximum(x*.01, x)

            else:
                x[self.act_inputs[layer]<= 0] = .01
                return x

        if act == 'softmax':
            if not der:
                exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_values / np.sum(exp_values, axis=1, keepdims=True)

            else:
                return x

    def Loss(self, y_pred, y_true, der=False):
        if self.loss == 'sparse_categorical_crossentropy' or self.loss == 'scce':
            # Number of samples in a batch
            samples = y_pred.shape[0]
            if not der:
                # Probabilities for target values - only if categorical labels
                if len(y_true.shape) == 1:
                    y_pred = y_pred[range(samples), y_true]
                # Losses
                negative_log_likelihoods = -np.log(y_pred)
                # Mask values - only for one-hot encoded labels
                if len(y_true.shape) == 2:
                    negative_log_likelihoods *= y_true
                # Overall loss
                return np.sum(negative_log_likelihoods) / samples
            else:
                y_pred[range(samples), y_true] -= 1
                y_pred /= samples
                return y_pred
        if self.loss == 'mean_absolute_error' or self.loss == 'mae': return np.sum(np.absolute(y_pred - y_true))
        if self.loss == 'mean_squared_error' or self.loss == 'mse': #L2
            losses = []
            for pred, true in zip(y_pred, y_true):
                losses.append(np.sum((pred - true)**2))
            return losses
        if self.loss == 'bce' or self.loss == 'binary_crossentrophy':
            raise NotImplementedError()
        if self.loss == 'cce' or self.loss == 'categorical_crossentrophy':
            raise NotImplementedError()

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

        # initialise delta weights & biases
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        
        # calculate derivitive loss
        X = self.Loss(X, Y, der=True)

        # backwards pass calculating derivitive values at each layer
        for i, a in reversed(list(enumerate(self.activations))):
            
            # derivitive loss of the activation functions
            X = self.Activate(a, X, der=True, layer=i)
            
            # delta values calculated here
            delta_weights[i] = np.dot(self.inputs[i].T, X)
            delta_biases[i] = np.sum(X, axis=0, keepdims=True)

            # set X for the next layer
            X = np.dot(X, self.weights[i].T)
        
        return delta_weights, delta_biases

    def validate(self):

        predictions = self.forward(self.X_valid)
        valid_loss = self.Loss(predictions, self.Y_valid)

        predictions = np.argmax(predictions, axis=1)
        targets = self.Y_valid

        valid_acc = np.mean(predictions == targets)
        return valid_loss, valid_acc

    def evaluate(self, data, target):

        print(f"\n\nTesting Set Evaluation:")
        print("-----------------------")

        predictions = self.forward(data)
        loss = self.Loss(predictions, target)

        predictions = np.argmax(predictions, axis=1)

        self.test_acc = np.mean(predictions == target)
        print(f"Test Acc:\t{np.sum(predictions == target)}/{target.shape[0]} ({self.test_acc:.2%})")
        print(f"Test Loss:\t{loss:.4f}\n")

    def predict(self, data):

        prediction = self.forward(data)
        score = np.amax(prediction)
        prediction = np.argmax(prediction)
        return prediction, score

    @staticmethod
    def Split(data, target, test_split=1/7, shuffle=True, seed=None):
       
        samples = data.shape[0]
        if shuffle:
            rp = np.random.RandomState(seed=seed).permutation(samples)
            data, target = data[rp], target[rp]
        test_size = int(samples * test_split)
        train_size = samples - test_size
        return data[:train_size], data[-test_size:], target[:train_size], target[-test_size:]

    @staticmethod
    def Encode(labels):
        count = 0
        output = []
        seen = {}
        code = {}
        total = set()
        for label in labels:
            if label in seen: output.append(seen[label])
            else:
                seen[label] = count
                output.append(seen[label])
                count += 1
                code[len(total)] = label
                total.add(label)
        return np.array(output), code

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