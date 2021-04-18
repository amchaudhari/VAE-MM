import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class SkipBlockDN(nn.Module):

	def __init__(self, in_ch, out_ch, skip_connections=True, downsample=False):

		"""
		Assumes that either of the two conditions are satisfied: 
			(i) the number of input channels (in_ch) and the number of output channels (out_ch) are equal;
			(ii) The number of output channels (out_ch) is exactly equal to twice the number of input channels (in_ch)
		"""
		assert in_ch==out_ch or in_ch==2*out_ch

		self.in_ch = in_ch
		self.out_ch = out_ch
		self.downsample = downsample
		self.skip_connections = skip_connections

		self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.bn2 = nn.BatchNorm2d(out_ch)
		if downsample:
			self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

	def forward(self, x):
		indentity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = F.relu(out)
		if self.in_ch == self.out_ch:
			out = self.convo2(out)
			out = self.bn2(out)
			out = F.relu(out)
		if self.downsample:
			out = self.downsampler(out)
			identity = self.downsampler(identity)
		if skip_connections:
			if self.in_ch == self.out_ch:
				out += identity
			else:
				out += identity[:,self.out_ch:,:,:]
		return out

class SkipBlockUP(nn.Module):

	def __init__(self, in_ch, out_ch, skip_connections=True, upscale=False):

		"""
		Assumes that either of the two conditions are satisfied: 
			(i) the number of input channels (in_ch) and the number of output channels (out_ch) are equal;
			(ii) The number of output channels (out_ch) is exactly equal to twice the number of input channels (in_ch)
		"""
		assert in_ch==out_ch or out_ch==2*in_ch

		self.in_ch = in_ch
		self.out_ch = out_ch
		self.upscale = upscale
		self.skip_connections = skip_connections

		self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=1, padding=1)
		self.conv2 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.bn2 = nn.BatchNorm2d(out_ch)
		if upscale:
			self.upscaler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, out_padding=1, padding=0)

	def forward(self, x):
		indentity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = F.relu(out)
		if self.in_ch == self.out_ch:
			out = self.convo2(out)
			out = self.bn2(out)
			out = F.relu(out)
		if self.upscale:
			out = self.upscaler(out)
			identity = self.upscaler(identity)
		if skip_connections:
			if self.in_ch == self.out_ch:
				out += identity
			else:
				out[:,:self.in_ch,:,:] += identity
				out[:,self.in_ch:,:,:] += identity
		return out


class skipVAE(nn.Module):

	def __init__(self, skip_connections=True, depth=16):

		super(VAE_skip_BN, self).__init__()
		if depth not in [8, 16, 32, 64]:
			sys.exit("VAE_skip_BN has been tested for depth for only 8, 16, 32, and 64")

		# self.z_dim = z_dim
		# self.hidden_dim = hidden_dim
		self.depth = depth//2

		#Encoder functions
		self.conv_in = nn.Conv2d(1, 64, 3, padding=1)
		self.pool = nn.MaxPool2d(2,2)
		self.bn1DN = nn.BatchNorm2d(64)
		self.bn2DN = nn.BATCHnORM2D(128)
		self.skip64DN_arr = nn.ModuleList()
		for i in range(self.depth):
			self.skip64DN_arr.append(SkipBlockDN(64, 64, skip_connections=skip_connections))
		self.skip64dsDN = SkipBlockDN(64, 64, downsample=True, skip_connections=skip_connections)
		self.skip64to128DN = SkipBlockDN(64, 128, skip_connections=skip_connections )
		self.skip128DN_arr = nn.ModuleList()
		for i in range(self.depth):
			self.skip128DN_arr.append(SkipBlockDN(128, 128, skip_connections=skip_connections))
		self.skip128dsDN1 = SkipBlockDN(128, 128, downsample=True, skip_connections=skip_connections)
		self.skip128dsDN2 = SkipBlockDN(128, 128, downsample=True, skip_connections=skip_connections)
		# self.fc1DN = nn.Linear(2048, hidden_dim)
		# self.fc2DN = nn.Linear(hidden_dim, z_dim)

		#Decoder functions
		self.bn1UP = nn.BatchNorm2d(128)
		self.bn2UP = nn.BatchNorm2d(64)
		self.skip64UP_arr = nn.ModuleList()
		for i in range(self.depth):
			self.skip64UP_arr.append(SkipBlockUP(64, 64, skip_connections=skip_connections))
		self.skip64dsUP = SkipBlockUP(64, 64, upscale=True, skip_connections=skip_connections)
		self.skip128dsUP = SkipBlockUP(128, 64, skip_connections=skip_connections)
		self.skip128UP_arr = nn.ModuleList()
		for i in range(self.depth):
			self.skip128UP_arr.append(SkipBlockUP(128, 128, skip_connections=skip_connections))
		self.skip128usUP = SkipBlockUP(128, 128, upscale=True, skip_connections=skip_connections)
		self.conv_out = nn.ConvTranspose2d(64, 1, 3, stride=2, dilation=2, output_padding=1, padding=2)
		
	def encoder(self, x):

		#Going down to the bottom of U
		x = self.pool(F.relu(self.conv_in(x)))
		for i, skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
			x = skip64(x)
		x = self.skip64dsDN(x)
		for i, skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
			x = skip64(x)
		x = self.bn1DN(x)
		x = self.skip64to128DN(x)
		for i, skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
			x = skip128(x)
		x = self.bn2DN(x)
		for i, skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
			x = skip128(x)
		z_mu = self.skip128dsDN1(x)
		z_log_sigma = self.skip128dsDN2(x)

		return z_mu, z_log_sigma

	def decoder(self, z):

		x = self.skip128usUP(z)
		for i, skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
			x = skip128(x)
		x = self.bn1UP(x)
		for i, skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
			x = skip128(x)
		x = self.skip128to64UP(x)
		for i, skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
			x = skip64(x)
		x = self.bn2UP(x)
		x = self.skip64usUP(x)
		for i, skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
			x = skip64(x)

		x_reconstr = F.sigmoid(self.conv_out(x))
		return x_reconstr		

	# Reconstruction + KL divergence losses summed over all elements and batch
	def loss_fn(self, x, x_reconstr, z_mu, z_log_sigma):
		BCE = F.binary_cross_entropy(x, x_reconstr, reduction='mean')
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5*torch.mean(1 + 2*z_log_sigma - z_mu**2 - torch.exp(z_log_sigma**2))

		return BCE + KLD

	def encoder_with_attributes(self, x, y):
		pass

	def decoder_with_attributes(self, z, y):
		pass