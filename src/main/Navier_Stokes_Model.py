
import os
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
project_path = os.environ.get('PROJECT_PATH')
data_path = os.environ.get('PROJECT_DATA_PATH')

data = scipy.io.loadmat(Path(data_path) / Path('cylinder_wake.mat'))

N_train = 5000

### STEP 1: PreProcessing ###

# U contains vx and vy (vector) for each time step and node (N x 2 x T)
U = data['U_star']  

# P contains p (scalar) for each time step and node (N x T)
P = data['p_star']  

# t is a 1-D array with the time steps (T x 1)
t = data['t']  

# X are the x/y coordinates of the mesh nodes (N x 2)
X = data['X_star']  

N = X.shape[0] # number of nodes
T = t.shape[0] # time steps

x_test = X[:, 0:1] # N x 1
y_test = X[:, 1:2] # N x 1
p_test = P[:, 0:1] # N x 1
u_test = U[:, 0:1, 0] # vx at time step 0 (N x 1)
t_test = np.ones((x_test.shape[0], x_test.shape[1])) # constant array of ones (N x 1)

# Rearrange Data
XX = np.tile(x_test, (1, T))  # N x T (for each coordinate, copy T time steps)
YY = np.tile(y_test, (1, T))  # N x T (same as for YY)
TT = np.tile(t, (1, N)).T  # N x T (N time series, one for each node)

UU = U[:, 0, :]  # vx for all nodes and time steps (N x T)
VV = U[:, 1, :]  # vy coordinates for all nodes and time steps (N x T)
PP = P  # pressure per node and time step N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

# Training Data (this approach allows us to use as much data as posible for training)
idx = np.random.choice(N * T, N_train, replace=False) #array with N random values within N*T
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]


# Define Model
class Navier_Stokes()