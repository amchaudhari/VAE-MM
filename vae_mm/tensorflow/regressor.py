import numpy as np
import tensorflow as tf
import keras

class Regressor(tf.keras.Model):
	def __init__(self):
		self.dense1 = Dense(100, activation="relu")
		self.dense2 = Dense(20, activation="relu")
		self.dense3 = Dense(3, activation="relu")

	def call(self, inputs):
		
		objs = self.dense1(inputs)
		objs = self.dense2(objs)
		objs = self.dense3(objs)

		return objs
		
