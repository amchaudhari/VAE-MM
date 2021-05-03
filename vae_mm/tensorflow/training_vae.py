import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from VAE import *
from utils import *

#Import data
df = pd.read_csv("saved/metamaterial_designs.csv")
df = df.fillna(method='ffill')
#Designs
X = df['image'].values
X = np.array([json.loads(x) for x in X])
#Attributes
Y = df[['vertical_lines', 'horizontal_lines', 'diagonals', 'triangles', 'three_stars']].values
Y = Y/lines[:,None]
means = np.mean(Y, axis=0, keepdims=True)
stds = np.std(Y, axis=0, keepdims=True)
Y = (Y-means)/stds
#Objectives
F = df[['obj1', 'obj2', 'constr1']].values

#Training parameters
W=28
H=28
latent_dim = 5				#The number of latent dimensions
cond_dim = 5				#The number of attributes
train_size = X.shape[0]
batch_size = 128
epochs = 100
beta = 5					#Weight for the KLE loss
gamma = 5					#Weight for the regression loss

#Split data for training and testing
# X_train, X_test, Y_train, Y_test, F_train, F_test = train_test_split(X, Y, F, test_size=0.33, random_state=42)


X_train = X_train.reshape(-1, W, H, 1).astype('float64')
X_test = X_test.reshape(-1, W, H, 1).astype('float64')

vae = VAE(latent_dim)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit([X, Y, F], epochs=150, batch_size=batch_size)
vae.save('./saved/vae2.h5')
