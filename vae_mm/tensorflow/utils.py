import numpy as np
import pandas as pd
from PIL import Image
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
# from imgaug import augmenters
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.gray()
# change the current directory to specified directory
from pathlib import Path
import os
p = Path(__file__).parents[2]
os.chdir(p)
import design_evaluator.python as truss_model
from constants import *
from showMeshPlot import showMeshPlot

def is_pareto_efficient(costs):
	"""
	Find the pareto-efficient points
	:param costs: An (n_points, n_costs) array
	:return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
	"""
	is_efficient = np.ones(costs.shape[0], dtype = bool)
	for i, c in enumerate(costs):
		is_efficient[i] = np.all(np.any(costs[:i]>=c, axis=1)) and np.all(np.any(costs[i+1:]>=c, axis=1))
	return is_efficient

def convert_to_img(x):
    plt.ioff()
    fig, ax = plt.subplots()
    x[x>0.25]=1
    x[x<0.25]=0
    showMeshPlot(nodes, edges, x, ifMatrix=False, ax=ax)
    fig.savefig('image.png', bbox_inches='tight')
    plt.close(fig)
    img = Image.open('image.png').convert('L').resize((28,28))
    x = np.array(img)<255
    
    return x.astype('int')

def plot_image(x, clean=True, ax=None):
	if ax is None:
		ax = plt.subplot()
	ax.imshow(1-x.squeeze())

def plot_1Dgrid_images(_images, image_size=28, figsize=3):
	"""
		_images: A list or an array of images
	"""
	n_img = len(_images)
	grid = np.linspace(0, n_img, n_img+1)
	figure = np.zeros((image_size * 1, image_size * n_img))
	for i in range(n_img):
		figure[ 0 : image_size, i * image_size : (i + 1) * image_size] = 1-_images[i]

	plt.figure(figsize=(figsize * n_img, figsize * 1))
	plt.imshow(figure, cmap="Greys_r")
	plt.show()