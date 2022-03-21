import torch
from typing import Union
import torch.nn as nn
import unittest
import numpy as np
import sys

class UNet_down(nn.Module):
	"""
	The 'down' blocks of the unet packaged into one class

	"""

	def __init__(self, input_size: int, output_size: int, shape, arg_padding, activation: str, batchnorm: bool):
		"""
		"""
		super(UNet_down, self).__init__()

		assert np.shape(shape) == (2,) or np.shape(shape) == ()
		if isinstance(shape, np.ndarray):
			assert len(shape) == 2 and shape.dtype == 'int'

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
			if np.shape(shape) == ():
				if shape % 2 == 0:
					low = (shape - 1) // 2
					high = (shape - 1) // 2 + 1
					self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
					arg_padding = 0
				else:
					arg_padding = (shape - 1) // 2
			else:
				if shape[0] == shape[1]: # shape must be an np.ndarray at this point
					shape = shape[0]
					if shape % 2 == 0:
						low = (shape - 1) // 2
						high = (shape - 1) // 2 + 1
						self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
						arg_padding = 0
					else:
						arg_padding = (shape - 1) // 2
				else: # shape[0] != shape[1]
					# shape size is (height, width)
					left_pad = (shape[1] - 1) // 2 
					right_pad = (shape[1] - 1) // 2 + 1 - (shape[1] % 2)
					top_pad = (shape[0] - 1) // 2 
					bottom_pad = (shape[0] - 1) // 2 + 1 - (shape[0] % 2)
					self.padding_layer = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))
					arg_padding = 0
		
		self.conv = nn.Conv2d(input_size, output_size, shape, stride=2, padding=arg_padding)	
		nn.init.kaiming_normal_(self.conv.weight)
		
		
	def forward(self, x):
		"""
		"""

		x = self.padding_layer(x)
		x = self.conv(x)
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

		assert np.shape(shape) == (2,) or np.shape(shape) == ()
		if isinstance(shape, np.ndarray):
			assert len(shape) == 2 and shape.dtype == 'int'

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
			if np.shape(shape) == ():
				if shape % 2 == 0:
					low = (shape - 1) // 2
					high = (shape - 1) // 2 + 1
					self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
					arg_padding = 0
				else:
					arg_padding = (shape - 1) // 2
			else:
				if shape[0] == shape[1]: # shape must be an np.ndarray at this point
					shape = shape[0]
					if shape % 2 == 0:
						low = (shape - 1) // 2
						high = (shape - 1) // 2 + 1
						self.padding_layer = nn.ZeroPad2d((low, high, low ,high))
						arg_padding = 0
					else:
						arg_padding = (shape - 1) // 2
				else: # shape[0] != shape[1]
					# shape size is (height, width)
					left_pad = (shape[1] - 1) // 2 
					right_pad = (shape[1] - 1) // 2 + 1 - (shape[1] % 2)
					top_pad = (shape[0] - 1) // 2 
					bottom_pad = (shape[0] - 1) // 2 + 1 - (shape[0] % 2)
					self.padding_layer = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))
					arg_padding = 0

		
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.conv = nn.Conv2d(input_size, output_size, shape, padding=arg_padding)	
		nn.init.kaiming_normal_(self.conv.weight)
		
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

class Head2plus1D(nn.Module):
	"""
	this is a 2+1D sepearated convolution
	The goal is to go from a 3D image (with temporal depth) to a 2d feature image
	this acts as a head for the UNet
	"""
	def __init__(self, in_planes, out_planes, mid_planes, depth_in):
		super(Head2plus1D, self).__init__()
		self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
		self.batchnorm1 = nn.BatchNorm3d(mid_planes)
		self.activation = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv3d(mid_planes, out_planes, kernel_size=(depth_in, 1, 1), stride=1, bias=False)
		self.batchnorm2 = nn.BatchNorm3d(out_planes)

	def forward(self, x):
		print("doing head")
		print("size before:", x.size())
		x = self.conv1(x)
		print("size after conv1:", x.size())
		x = self.batchnorm1(x)
		print("size after batchnorm1:", x.size())
		x = self.activation(x)
		print("size after activ:", x.size())
		x = self.conv2(x)
		print("size after conv2:", x.size())
		x = self.batchnorm2(x)
		x = self.activation(x)
		return torch.squeeze(x, dim=2)