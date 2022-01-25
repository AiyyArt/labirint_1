"""
**************************************

Created in 21.01.2022
by Aiyyskhan Alekseev

https://github.com/AiyyArt
timirkhan@gmail.com

**************************************
"""

__author__ = "Aiyyskhan Alekseev"
__version__ = "2.2.0"

import numpy as np
import torch

DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Ganglion_torch:
	"""
	Класс нейросетевого ганглиона
	"""
	def __init__(self, weight_list):
		self.W0 = torch.tensor(weight_list[0], dtype=DTYPE, device=DEVICE)
		self.W1 = torch.tensor(weight_list[1], dtype=DTYPE, device=DEVICE)
		self.W2 = torch.tensor(weight_list[2], dtype=DTYPE, device=DEVICE)

	def __call__(self, input_data): #, w_mode=True):
		# матрица весов или матрица типов
		input_data = torch.tensor(input_data[:,None,:], dtype=DTYPE, device=DEVICE)
		
		potentials = torch.relu(torch.tanh(torch.matmul(input_data, self.W0)))

		potentials = torch.relu(torch.tanh(torch.matmul(potentials, self.W1)))
		
		output = torch.relu(torch.tanh(torch.matmul(potentials, self.W2)))

		return torch.reshape(output, (output.shape[0], output.shape[2])).cpu().numpy()

class Ganglion_numpy:
	"""
	Класс нейросетевого ганглиона
	"""
	def __init__(self, weight_list):
		self.W0 = weight_list[0]
		self.W1 = weight_list[1]
		self.W2 = weight_list[2]

		self.relu = lambda x: (np.absolute(x) + x) / 2

	def __call__(self, input_data):
		# матрица весов или матрица типов
		
		potentials = self.relu(np.tanh(np.matmul(input_data, self.W0)))

		potentials = self.relu(np.tanh(np.matmul(potentials, self.W1)))
		
		output = self.relu(np.tanh(np.matmul(potentials, self.W2)))

		return output