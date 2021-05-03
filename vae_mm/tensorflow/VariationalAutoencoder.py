import tensorflow as tf
from keras import backend as K

class VAE(tf.keras.Model):
	def __init__(self,
			latent_dim,
			beta=1,
			gamma=1,
			use_regressor=False,
			name="VariationalAutoencoder",
			**kwargs
		):
		super(VAE, self).__init__(**kwargs)

		self.latent_dim = latent_dim
		self.beta = beta
		self.gamma = gamma
		self.encoder = Encoder(latent_dim=latent_dim)
		self.decoder = Decoder()
		self.regressor = Regressor()
		self.use_regressor = use_regressor

	def train_step(self, data):
		if isinstance(data, tuple):
			data = data[0]
		x = data[0]
		y = data[1]
		if self.use_regressor:
			f = data[2]
		n_sample = tf.shape(x)[0]
		n_objs = tf.shape(f)[0]
		
		with tf.GradientTape() as tape:
			y_ = tf.expand_dims(tf.expand_dims(y, 1), 1)
			k = tf.ones([n_sample, 28, 28, 1])
			h = tf.concat([x, k*y_], 3)
			z_mean, z_log_var, z = self.encoder(h)
			
			h = tf.concat([z, y], 1)
			reconstruction = self.decoder(h)
			
			regression_loss = 0
			if self.use_regressor:
				f_pred = self.regressor(h)
				regression_loss = tf.reduce_mean(
					tf.keras.losses.mean_squared_error(f, f_pred)
				)
				regression_loss *= tf.cast(n_objs, tf.float32)
			
			reconstruction_loss = tf.reduce_mean(
				tf.keras.losses.binary_crossentropy(x, reconstruction)
			)
			reconstruction_loss *= 28 * 28
			tf.keras.layers_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
			tf.keras.layers_loss = tf.reduce_mean(tf.keras.layers_loss)
			tf.keras.layers_loss *= -0.5
			total_loss = reconstruction_loss + self.beta*tf.keras.layers_loss + self.gamma*regression_loss
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		
		return {
			"loss": total_loss,
			"reconstruction_loss": reconstruction_loss,
			"tf.keras.layers_loss": tf.keras.layers_loss,
			"regression_loss": regression_loss
		}
#####################################################################################
## Encoder Network
#####################################################################################

class Encoder(tf.keras.Model):
	
	def __init__(self, latent_dim, name="encoder"):
		super(Encoder, self).__init__(name=name)
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
		self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(16, activation="relu")
		self.dense21 = tf.keras.layers.Dense(latent_dim, name="z_mu")
		self.dense22 = tf.keras.layers.Dense(latent_dim, name="z_log_sigma")
		self.sample = Sampling()

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
class Sampling(tf.keras.layers.Layer):
	"""Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = K.random_normal(shape=(batch, dim))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#####################################################################################
## Decoder Network
#####################################################################################
class Decoder(tf.keras.Model):

	def __init__(self, name="decoder"):
		super(Decoder, self).__init__(name=name)
		self.dense = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")
		self.reshape = tf.keras.layers.Reshape((7, 7, 64))
		self.conv1 = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
		self.conv2 = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
		self.conv3 = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

	def call(self, inputs):
		x = self.dense(inputs)
		x = self.reshape(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x_reconstructred = self.conv3(x)

		return x_reconstructred

#####################################################################################
## Regressor Network
#####################################################################################
class Regressor(tf.keras.Model):
	def __init__(self, name="regressor"):
		super(Regressor, self).__init__(name=name)
		self.dense1 = tf.keras.layers.Dense(100, activation="relu")
		self.dense2 = tf.keras.layers.Dense(20, activation="relu")
		self.dense3 = tf.keras.layers.Dense(3, activation="relu")

	def call(self, inputs):
		
		objs = self.dense1(inputs)
		objs = self.dense2(objs)
		objs = self.dense3(objs)

		return objs

#####################################################################################
## Bitstring Decoder Network (Converts an image into a bitstring) 
#####################################################################################
class DecoderBitstring(tf.keras.Model):

	def __init__(self, num_bits, name="decoder bitstring"):
		self.conv1 = Conv2D(28, kernel_size=(3, 3), activation="softplus", strides=2, padding="same")
		self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
		self.conv2 = Conv2D(56, kernel_size=(3, 3), activation="softplus", strides=2, padding="same")
		self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
		self.flat = Flatten()
		self.dropout = Dropout(0.5)
		self.dense = Dense(num_bits, activation="sigmoid")

	def call(self, inputs):
		x = self.maxpool1(self.conv1(x))
		x = self.maxpool2(self.conv2(x))
		x = self.dropout(self.flat(x))
		x = self.dense(x)
		return x