import torch
import torch.nn as nn

import unittest

import numpy as np

class UNet(nn.Module):
	def __init__(self, input_size, num_classes, filter_multiplier=10, regularization_rate=0.):
		super(UNet, self).__init__()
		self.input_size = input_size
		self.num_classes = num_classes
		self.filter_multiplier = filter_multiplier
		self.regularization_rate = regularization_rate

		self.num_layers = int(np.floor(np.log2(input_size)))
		self.down_conv_kernel_sizes = np.zeros([num_layers], dtype=int)	
		self.down_filter_numbers = np.zeros([num_layers], dtype=int)
		self.up_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
		self.up_filter_numbers = np.zeros([num_layers], dtype=int)

		for layer_index in range(self.num_layers):
			self.down_conv_kernel_sizes[layer_index] = int(3)
			self.down_filter_numbers[layer_index] = int((layer_index + 1) * self.filter_multiplier + self.num_classes)
			self.up_conv_kernel_sizes[layer_index] = int(4)
			self.up_filter_numbers[layer_index] = int((self.num_layers - layer_index - 1) * self.filter_multiplier + self.num_classes)


	def forward(self, x):
		skips = []

		for shape, filters in zip(self.down_conv_kernel_sizes, self.down_filter_numbers):
			skips.append(x)
			x = nn.Conv2D(filters, (shape, shape), strides=2, padding="same", activation="relu",
			bias_regularizer=l1(self.regularization_rate))(x)

		for shape, filters in zip(self.up_conv_kernel_sizes, self.up_filter_numbers):
			x = nn.UpSampling2D()(x)
			skip_output = skips.pop()
			x = concatenate([x, skip_output], axis=3)
			if filters != self.num_classes:
				x = nn.Conv2D(filters, (shape, shape), activation="relu", padding="same",
				bias_regularizer=l1(rself.egularization_rate))(x)
				x = nn.BatchNormalization(momentum=.9)(x)
			else:
				x = nn.Conv2D(filters, (shape, shape), activation="softmax", padding="same",
				bias_regularizer=l1(self.regularization_rate))(x)

		assert len(skips) == 0
