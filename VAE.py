import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):

	def __init__(self, z_dim, hidden_dim):

		super(VAE, self).__init__()

		self.z_dim = z_dim
		self.hidden_dim = hidden_dim

		self.conv2d1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
		self.conv2d2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
		
		# self.fc1 = nn.Linear(784, hidden_dim)
		self.fc21 = nn.Linear(784, z_dim)
		self.fc22 = nn.Linear(784, z_dim)

		self.fc3 = nn.Linear(z_dim, 784)
		self.conv2d3 = nn.ConvTranspose2d(16, 16, 3, stride=2, output_padding=1, padding=1)
		self.conv2d4 = nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1, padding=1)
		self.conv2d5 = nn.ConvTranspose2d(8, 1, 3, padding=1, dilation=1)

	def encoder(self, x):

		x = F.relu(self.conv2d1(x))
		x = F.relu(self.conv2d2(x))
		x = torch.flatten(x, start_dim=-3, end_dim=-1)
		# x = F.relu(self.fc1(x))

		z_mu = self.fc21(x)
		z_log_var = self.fc22(x)

		return z_mu, z_log_var

	def decoder(self, z):

		x = F.relu(self.fc3(z))
		batch_size = x.shape[0]
		x = x.reshape((batch_size,16,7,7))

		x = F.relu(self.conv2d3(x))
		x = F.relu(self.conv2d4(x))
		x_reconstr = F.sigmoid(self.conv2d5(x))

		return x_reconstr		

	def forward(self, x, **kwargs):
		mu, log_var = self.encoder(x)
		z = torch.normal(mu, torch.exp(0.5*log_var))
		return  [self.decoder(z), mu, log_var]

	# Reconstruction + KL divergence losses summed over all elements and batch
	def loss_fn(self, x, x_reconstr, z_mu, z_log_var):
		BCE = F.binary_cross_entropy(x_reconstr, x, reduction='mean')
		BCE *= 28*28
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5*torch.mean(1 + z_log_var - z_mu**2 - torch.exp(z_log_var))

		return {'loss': BCE+KLD, 'BCELoss':BCE, 'KLDLoss':KLD}

	def encoder_with_attributes(self, x, y):
		pass

	def decoder_with_attributes(self, z, y):
		pass