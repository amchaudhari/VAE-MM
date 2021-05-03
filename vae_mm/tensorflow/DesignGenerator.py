import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import *

class DesignGenerator(object):

	def __init__(self, encoder, decoder, regressor, decoder_bistring, image_size=28):
		self.encoder = encoder
		self.decoder = decoder
		self.decoder_bistring = decoder_bistring
		self.regressor = regressor
		self.image_size = image_size
		# self.latent_dim = latent_dim

	def encoder_input(self, _image, _attr):
		x_input = _image.reshape(1, self.image_size, self.image_size,1)
		if len(_attr.shape)==1:
			_attr = _attr[None,:]
		n_attr = _attr.shape[1]
		y_ = _attr[None,:].reshape(-1,1,1,n_attr)
		n_sample = x_input.shape[0]
		k = tf.ones([n_sample, 28, 28, 1])
		x_input = tf.concat([x_input, k*y_], 3)
		return x_input

	def get_driving_features(self, _images, _attr=None, K=10):
		"""
			Function to find driving features common in given set of images
			_images: A list or array of NxN images in numpy format
			_attr: A list or array of (N,a) sized attributes of given images
			K: Number of top driving features
			
		"""
		features = list()
		if len(_attr.shape)==1:
			_attr = _attr[None, :]

		n_attributes = _attr.shape[1]
		for i in range(len(_images)):
			x = _images[i].reshape(1, self.image_size, self.image_size,1)
			x_input = self.encoder_input(x, _attr[i])
			features.append(self.encoder(x_input)[-1])
		features = np.array(features).squeeze()
		
		if np.all(_attr):
			h = tf.concat([features, _attr], 1)
		else:
			h = features
		kmeans = KMeans(n_clusters=K, random_state=0).fit(h)
		cluster_means = kmeans.cluster_centers_

		_reconstr_images = list()
		for mean in cluster_means:
			_reconstr_images.append(self.decoder(mean[None,:]))
			
		_reconstr_images = np.array(_reconstr_images).squeeze()

		return _reconstr_images

	def superposition_attributes(self, _image, _attr, **kwargs):

		diff = np.array(list(kwargs.values()))

		new_attr = _attr[None, :] + diff[None, :]

		x_input = self.encoder_input(_image, _attr)
		features = self.encoder(x_input)[-1]
		
		h = tf.concat([features, new_attr], 1)
		_reconstr_image = self.decoder(h)
		_reconstr_image_bits = self.decoder_bistring.predict(_reconstr_image)
		_reconstr_image = convert_to_img(np.squeeze(_reconstr_image_bits))
		
		return _reconstr_image

	def superposition_features(self, _image, _attr, **kwargs):

		diff = np.array(list(kwargs.values()))

		x_input = self.encoder_input(_image, _attr)
		features = self.encoder(x_input)

		new_features = features + diff[None, :]

		h = tf.concat([new_features, _attr[None, :]], 1)
		_reconstr_image = self.decoder(h)
		_reconstr_image_bits = self.decoder_bistring.predict(_reconstr_image)
		_reconstr_image = convert_to_img(np.squeeze(_reconstr_image_bits))
		
		return _reconstr_image

	def autosuperposition(self, _image, _attr, eta, ref_point, if_plot=False):

		# ref_point = tf.constant([[1., 0., 1.]])
		cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
		x_input = self.encoder_input(_image, _attr)
		z = self.encoder(x_input)[-1]
		_attr = tf.constant(_attr, dtype="float32")
		h = tf.concat([z, _attr[None,:]], 1)
		with tf.GradientTape() as tape:
			tape.watch(h)	
			f_pred = self.regressor(h)
			dummy_loss = 10*cosine_loss(ref_point[:2], f_pred[0:2])
		
		dloss_dh = tape.gradient(dummy_loss, h)
		delta_h = h*dloss_dh
		h_new = h-eta*delta_h

		if if_plot:
			print("old z:", np.around(h,2), "new z:", np.around(h_new.numpy(),2))
			print("old f:", self.regressor(h).numpy(), "new f:", self.regressor(h_new).numpy())
		
		# h_new = tf.concat([z_new, _attr[None,:]], 1)
		_reconstr_image = self.decoder(h_new)
		# _reconstr_image_bits = self.decoder_bistring.predict(_reconstr_image)
		# _reconstr_image = convert_to_img(np.squeeze(_reconstr_image_bits))
		
		return _reconstr_image
