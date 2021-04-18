from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE import *
from skipVAE import *
from metamaterial_design_dataset import *
from torchsummary import summary

parser = argparse.ArgumentParser(description='VAE Mechanical Metamaterial Example')
parser.add_argument('--root_path', required =True, type =str,
					help='root path for the dataset')
parser.add_argument('--csv_file', required =True , type = str,
					help='csv file name for the dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--skip-connections', type=bool, default=False, metavar='N',
					help='Whether to implement skip connections')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = metamaterial_design_dataset(args.csv_file, args.root_path)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)


z_dim = 5
hidden_dim = 100
if args.skip_connections:
	model = skipVAE().to(device)
else:
	model = VAE(z_dim, hidden_dim).to(device)

summary(model, (1,28,28))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
	model.train()
	epoch_loss = 0

	for batch_idx, sample in enumerate(train_data_loader):
		x = sample['image'].float()
		x = x.to(device)
		x_reconstr, z_mu, z_log_var = model.forward(x)

		if torch.any(torch.isnan(x_reconstr)):
			print(x_reconstr, z_mu, z_log_var)

		ret = model.loss_fn(x, x_reconstr, z_mu, z_log_var)
		train_loss = ret['loss']
		train_loss.backward()
		optimizer.step()
		torch.autograd.set_detect_anomaly(True)

		epoch_loss += train_loss.item()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotalLoss: {:.6f}\tBCELoss: {:.6f}\tKLDLoss: {:.6f}'.format(
				epoch, batch_idx * len(x), len(train_data_loader.dataset),
				100. * batch_idx / len(train_data_loader),
				train_loss.item() / len(x),
				ret['BCELoss'].item() / len(x),
				ret['KLDLoss'].item() / len(x)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, epoch_loss / len(train_data_loader.dataset)))


def test(epoch):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for i, (x, _) in enumerate(test_loader):
			x = x.to(device)
			z_mu, z_log_sigma = self.encoder(x.float())
			z = torch.normal(z_mu, torch.exp(tz_log_sigma))
			x_reconstr = self.decoder(z)

			test_loss += model.loss_fn(x, x_reconstr, z_mu, z_log_sigma).item()
			if i == 0:
				n = min(x.size(0), 8)
				comparison = torch.cat([x[:n],
									  x_reconstr.view(args.batch_size, 1, 28, 28)[:n]])
				save_image(comparison.cpu(),
						 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
	for epoch in range(1, args.epochs + 1):
		train(epoch)
		# test(epoch)
		with torch.no_grad():
			sample = torch.randn(64, z_dim).to(device)
			sample = model.decoder(sample).cpu()
			save_image(sample.view(64, 1, 28, 28),
					   'results/sample_' + str(epoch) + '.png')




