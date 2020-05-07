import numpy as np
np.random.seed(0)

class NeuralNetwork:

	def __init__(self):
		self.inputL = 0
		self.outputL = 0
		self.hlayers = 0
		self.input_size = 0
		self.weights = []
		self.biases = []
		self.params = 0
		self.samples = 0
		self.features = 0

	def add(self, layer=None, data=None, n_neurons=None, transform=None, value=None):

		if layer not in ('input', 'hidden', 'output', 'transform'): 
			raise ValueError(f"'{layer}' is not a valid layer. Try 'input', 'hidden', 'output' or 'transform'.")

		if layer == 'input':
			if isinstance(data, np.ndarray) == False:
				raise TypeError(f'{type(self.data)} must be a numpy array.')

			if len(data.shape) != 2:
				raise ValueError(f'bad input shape {data.shape}')

			self.data = data
			self.input_size = data.shape[1]
			self.samples = self.data.shape[0]
			self.features = self.data.shape[1]
			self.inputL += 1

		if layer == 'hidden':
			self.hlayers += 1

			if self.hlayers == 1:
				self.n_input_size = self.input_size
			else:
				self.n_input_size = self.previousLayerSize
			self.params += self.n_input_size * n_neurons ##PARAMS
			self.weights.append(0.10 * np.random.randn(self.n_input_size, n_neurons))
			self.biases.append(np.zeros(n_neurons))
			self.params += n_neurons ##PARAMS
			self.previousLayerSize = n_neurons

		if layer == 'output':
			self.outputL += 1
			self.output_size = len(np.unique(data))
			self.params += self.previousLayerSize * self.output_size ##PARAMS
			self.output_weights = 0.10 * np.random.randn(self.previousLayerSize, self.output_size)
			self.output_biases = np.zeros(self.output_size)
			self.params += self.output_size #PARAMS
		if layer == 'transform':
			if self.hlayers != 0:
				raise RuntimeError("All transform layers must be added before any hidden layers.")
			if transform not in ('normalize', 'standardize'):
				raise ValueError(f"{transform} is not a valid transform. Try 'normalize' or 'standardize'.")
			elif transform == 'normalize':
				xmax = np.amax(self.data)
				xmin = np.amin(self.data)
				normalize = lambda x: (x-xmin) / (xmax - xmin)
				self.data = normalize(self.data)
			elif transform == 'standardize':
				xmean = np.mean(self.data)
				xstd = np.std(self.data)
				standardize = lambda x: (x-xmean) / xstd
				self.data = standardize(self.data)
				print(self.data)

	def summary(self):
		print('Neural Network implemented with the following structure:\n')
		print(f'Input: {self.data.shape[0]} samples each with {self.data.shape[1]} features.')
		for ilayer, n in enumerate(self.weights):
			print(f'[{ilayer+1}] Hidden: {n.shape[1]} neurons.')
		print(f'Output: {self.output_size} classses.')

		print('--------------------------')
		print(f'Total parameters: {self.params:,}')
		print('--------------------------\n')

	def compile(self):
		if self.hlayers < 1:
			raise NotImplementedError("No hidden layers detected. Use add(layer='hidden') to add a hidden layer.")
		if self.inputL == 0:
			raise NotImplementedError("No input layer detected. Use add(layer='input') to add an input layer.")
		if self.outputL == 0:
			raise NotImplementedError("No output layer detected. Use add(layer='output') to add an output layer.")
		if self.inputL > 1:
			raise RuntimeError("More than 1 input layer detected.")
		if self.outputL > 1:
			raise RuntimeError("More than 1 output layer detected.")

	def fit(self):
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
		print('Predictions:')
		print('------------')
		print(self.output)

	def evaluate(self):
		pass