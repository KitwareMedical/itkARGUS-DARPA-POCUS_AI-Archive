import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
import unittest
import numpy as np
import sys

class UNet_down(nn.Module):
	"""
	The 'down' blocks of the unet packaged into one class
	"""

	def __init__(self, input_size: int, output_size: int, shape: int, arg_padding, activation: str, batchnorm: bool):
		"""
		"""
		super(UNet_down, self).__init__()

		self.padding_layer = nn.Identity()

		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'softmax':
			self.activation = nn.Softmax()
		else:
			sys.exit("activation must be 'relu' or 'softmax'")

		if batchnorm:
			self.batchnorm = nn.BatchNorm2d(output_size)
		else:
			self.batchnorm = nn.Identity()

		if arg_padding == 'same':
			if shape % 2 == 0:
				low = (shape - 1) // 2
				high = (shape - 1) // 2 + 1
				self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
				arg_padding = 0
			else:
				arg_padding = (shape - 1) // 2
		
		self.conv = nn.Conv2d(input_size, output_size, shape, stride=2, padding=arg_padding)	
		
		
	def forward(self, x):
		"""
		"""

		x = self.conv(x)
		x = self.padding_layer(x)
		x = self.activation(x)
		x = self.batchnorm(x)

		return x

class UNet_up(nn.Module):
	"""
	The 'up' blocks of our unet packaged into one class
	"""

	def __init__(self, input_size: int, output_size: int, shape: int, arg_padding, activation, batchnorm: bool):
		"""
		"""
		super(UNet_up, self).__init__()

		self.padding_layer = nn.Identity()

		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'softmax':
			self.activation = nn.Softmax()
		elif activation == None:
			self.activation = nn.Identity()
		else:
			sys.exit("activation must be 'relu', 'softmax', or None")

		if batchnorm:
			self.batchnorm = nn.BatchNorm2d(output_size)
		else:
			self.batchnorm = nn.Identity()

		if arg_padding == 'same':
			if shape % 2 == 0:
				low = (shape - 1) // 2
				high = (shape - 1) // 2 + 1
				self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
				arg_padding = 0
			else:
				arg_padding = (shape - 1) // 2
		
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.conv = nn.Conv2d(input_size, output_size, shape, padding=arg_padding)	
		
	def forward(self, x, skip):
		"""
		"""

		x = self.upsample(x)
		x = torch.cat([x, skip], axis=1)
		x = self.padding_layer(x)
		x = self.conv(x)
		x = self.activation(x)
		x = self.batchnorm(x)

		return x