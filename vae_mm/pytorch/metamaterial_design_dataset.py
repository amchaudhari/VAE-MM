from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

class metamaterial_design_dataset(Dataset):
	"""Metamaterial designs dataset."""

	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		file_path = os.path.join(root_dir, csv_file)
		self.designs = pd.read_csv(file_path)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.designs)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		df = self.designs
		image = torch.tensor(json.loads(df.iloc[idx, df.columns.get_loc('image')])).unsqueeze(axis=0)
		col_idx = [df.columns.get_loc(col) for col in ['obj1', 'obj2', 'constr1']]
		outputs = torch.tensor(df.iloc[idx, col_idx].to_numpy(dtype=np.float32))
		col_idx = [df.columns.get_loc(col) for col in ['vertical_lines', 'horizontal_lines', 'diagonals', 'triangles', 'three_stars']]
		attributes = torch.tensor(df.iloc[idx, col_idx].to_numpy(dtype=np.float32))
		sample = {'image':image, 'outputs':outputs, 'attributes':attributes}

		if self.transform:
			sample['image'] = self.transform(image)

		return sample