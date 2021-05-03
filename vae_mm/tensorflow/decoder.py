import numpy as np
import tensorflow as tf
import keras
from keras.layers import *

class Decoder(tf.keras.Model):

	def __init__(self):
		self.dense = Dense(7 * 7 * 64, activation="relu")
		self.reshape = Reshape((7, 7, 64))
		self.conv1 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
		self.conv2 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
		self.conv3 = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

	def call(self, inputs):
		x = self.dense(inputs)
		x = self.reshape(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x_reconstructred = self.conv3(x)

		return x_reconstructred
