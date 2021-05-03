import numpy as np
import tensorflow as tf
import keras
from keras.layers import *

class Encoder(tf.keras.Model):
	
	def __init__(self, latent_dim):
		
		self.latent_dim = latent_dim
		self.conv1 = Conv2D(32, 3, activation="relu", strides=2, padding="same")
		self.conv2 = Conv2D(64, 3, activation="relu", strides=2, padding="same")
		self.flat = Flatten()
		self.dense1 = Dense(16, activation="relu")
		self.dense21 = Dense(latent_dim, name="z_mu")
		self.dense22 = Dense(latent_dim, name="z_log_sigma")
		self.sample = self.Sampling()

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.flat(x)
		x = self.dense1(x)

		z_mu = self.dense21(x)
		z_log_sigma = self.dense22(x)
		z = self.sample([z_mu, z_log_sigma])

		return z_mu, z_log_sigma, z

	# A different autoencoder
	class Sampling(Layer):
	    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

	    def call(self, inputs):
	        z_mean, z_log_var = inputs
	        batch = tf.shape(z_mean)[0]
	        dim = tf.shape(z_mean)[1]
	        epsilon = K.random_normal(shape=(batch, dim))
	        return z_mean + tf.exp(0.5 * z_log_var) * epsilon