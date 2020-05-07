import numpy as np

class NeuralNetwork:

	def __init__(self):
		self.inputExists = False
		self.outputExists = False
		self.hlayers = 0
		self.input_size = 0
		self.weights = []
		self.biases = []
		self.params_count = 0
		self.samples = 0
		self.transforms_list = []
		self.test_size = 0
		self.has_split = False
		self.flattened = False
		self.x_transform = False
		self.y_transform = False 
	
	def input(self, data=None, labels=None, flatten=False):

		if data is None:
			raise ValueError("No training data passed. Use data=X to input training data")

		if labels is None:
			raise ValueError("No label data passed. Use labels=Y to input label data")

		if not isinstance(data, np.ndarray):
			raise TypeError(f'{type(data)} must be a numpy array')

		if not isinstance(labels, np.ndarray):
			raise TypeError(f'{type(labels)} must be a numpy array')
		
		if flatten == True:
			data = data.reshape(data.shape[0], -1)
			self.flattened = True
		else:
			if len(data.shape) > 2:
				data = data.reshape(data.shape[0], -1)
			self.flattened = True
			

		if len(data.shape) != 2:
			raise ValueError(f'Bad training data input shape {data.shape}. Try flattening the data with flatten=True')

		self.inputExists = True
		self.data = data
		self.labels = labels
		self.input_size = data.shape[1]
		self.output_size = len(np.unique(self.labels))
		self.samples = data.shape[0]

	def transform(self, X=True, Y=False, transform='normalize'):

		self.inputExistCheck()
		if self.hlayers != 0:
			raise RuntimeError("All transforms must be done before you add any hidden layers.")
		if transform not in ('normalize', 'standardize', 'categorical'):
			raise ValueError(f"{transform} is not a valid transform. Try 'normalize', 'standardize' or 'categorical'.")

		transformed = False

		if X and transform == 'normalize':
			xmax = np.amax(self.data)
			xmin = np.amin(self.data)
			normalize = lambda x: (x-xmin) / (xmax - xmin)
			self.data = normalize(self.data)
			self.transforms_list.append(('Normalize', X, Y))
			self.x_transform = True
			transformed = True

		if X and transform == 'standardize':
			xmean = np.mean(self.data)
			xstd = np.std(self.data)
			standardize = lambda x: (x-xmean) / xstd
			self.data = standardize(self.data)
			self.transforms_list.append(('Standardize', X, Y))
			self.x_transform = True
			transformed = True

		if Y and transform == 'categorical':
			self.labels = np.eye(self.output_size)[self.labels]
			self.transforms_list.append(('Categorical', X, Y))
			self.y_transform = True
			transformed = True

		if not transformed:
			raise Warning(f'No transform completed as {transform} is incompatible with this data.')

	def split(self, test_split=1/7, shuffle=True, random_state=0):
		self.inputExistCheck()
		np.random.seed(random_state)

		try:
			self.test_size = int(self.samples*test_split)
			self.data, self.data_test = self.data[:-self.test_size], self.data[-self.test_size:]
			self.labels, self.labels_test = self.labels[:-self.test_size], self.labels[-self.test_size:]
			self.has_split = True
		except:
			raise ValueError(f'{test_split} is not a valid train/test split. Default is 0.2.')

	def addLayer(self, n_neurons=128, activation='relu'):

			self.splitCheck()

			self.hlayers += 1

			if self.hlayers == 1:
				self.n_input_size = self.input_size
			else:
				self.n_input_size = self.previousLayerSize

			self.params_count += self.n_input_size * n_neurons
			self.weights.append(0.10 * np.random.randn(self.n_input_size, n_neurons))
			self.biases.append(np.zeros(n_neurons))
			self.params_count += n_neurons
			self.previousLayerSize = n_neurons

	def output(self, activation='softmax'):
			
			if self.hlayers < 1:
				raise NotImplementedError("No hidden layer detected. Use add() to add a hidden layer.")

			self.outputExists = True
			self.params_count += self.previousLayerSize * self.output_size
			self.output_weights = 0.10 * np.random.randn(self.previousLayerSize, self.output_size)
			self.output_biases = np.zeros(self.output_size)
			self.params_count += self.output_size

	def compile(self, valid_split=0.2, optimizer='adam', batch_size=128, epochs=10):
		self.outputExistCheck()

	def summary(self):
		print('Function/Layer [Shape]')
		print('-----------------------')
		if self.flattened:
			print(f'Input [{self.samples}, {int(self.input_size**0.5)}, {int(self.input_size**0.5)}]')
			print('\t   |   ')
			print(f'Flatten [{self.samples}, {self.input_size}]')
			print('\t   |   ')
		else:
			print(f'Input [{self.samples}, {self.input_size}]')
			print('\t   |   ')
		if self.x_transform:
			print(f'Normalize [{self.samples}, {self.input_size}]')
			print('\t   |   ')
		print(f'Split [{self.data.shape[0]}, {self.input_size}]')
		print('\t   |   ')
		print('\tTraining')
		print('\t   |   ')
		for n in self.weights:
			print(f'Hidden [{self.data.shape[0]}, {n.shape[1]}]\n\t   |   ')
		print(f'Output [{self.data.shape[0]}, {self.output_size}]')
		print('--------------------------')
		print(f'Total parameters: {self.params_count:,}')
		print('--------------------------\n')
		print(f'Predict [{self.data_test.shape[0]}, {self.input_size}]')
		print('\t   |   ')
		print(f'Arg Max [{self.data_test.shape[0]}, {self.output_size}]')
		print('\t   |   ')
		print(f'Predictions [{self.data_test.shape[0]}, 1]\n')

	def train(self):
		# 1 hidden layer only
		if self.hlayers == 1: 					
			self.last_output = np.dot(self.data, self.weights[0]) + self.biases[0]
		# multiple hidden layers	
		else:
			for layer_num in range(len(self.weights)):
				# skip on first layer iteration (last_output not set yet)
				if layer_num != 0: 								
					self.data = self.last_output

				self.last_output = np.dot(self.data, self.weights[layer_num]) + self.biases[layer_num]

		self.output = np.dot(self.last_output, self.output_weights) + self.output_biases

	def predict(self):
		print(f'Predicting on {self.test_size:,} samples...')
		print('-------------------------------')
		print(self.output)

	def evaluate(self):
		pass

	def splitCheck(self):
		if not self.has_split:
			raise NotImplementedError(f'You have not split the data into training/test yet. Use split() before implementing hidden layers.')

	def inputExistCheck(self):
		if not self.inputExists:
			raise NotImplementedError("Model has no input yet. Use input(X, Y) to add input.")

	def outputExistCheck(self):
		if not self.outputExists:
			raise NotImplementedError("Model has no output yet. Use output() to add output.")

