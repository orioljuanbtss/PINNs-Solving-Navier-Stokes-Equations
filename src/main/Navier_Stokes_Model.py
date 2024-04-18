
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

# Load data
data = scipy.io.loadmat(Path(data_path) / Path('cylinder_wake.mat'))


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

#perform 90-10 sampling 
N_train = int(0.9 * N * T)
N_test = N * T - N_train

train_idx = np.random.choice(N * T, N_train, replace=False)
test_idx = np.setdiff1d(np.arange(N * T), train_idx) #indices not intersecting train


# Rearrange Data
XX = np.tile(X[:, 0:1], (1, T))  # N x T (for each coordinate, copy T time steps)
YY = np.tile(X[:, 1:2], (1, T))  # N x T (same as for YY)
TT = np.tile(t, (1, N)).T  # N x T (N time series, one for each node)

UU = U[:, 0, :]  # vx for all nodes and time steps (N x T)
VV = U[:, 1, :]  # vy coordinates for all nodes and time steps (N x T)
PP = P  # pressure per node and time step N x T


# Each node's data (x, y, t, u, v) needs to be associated with each time step. 
x = XX.flatten()[:, None]  # NT x 1 (x0 at t= 0, 0 at t = 1, x0 at t = 2...)
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1


# Training data
x_train, y_train, t_train, u_train, v_train = x[train_idx], y[train_idx], t[train_idx], u[train_idx], v[train_idx]

# Testing data
x_test, y_test, t_test, u_test, v_test = x[test_idx], y[test_idx], t[test_idx], u[test_idx], v[test_idx]


# Define custom NN

class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        #find best architecture
        self.layers = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2)
        )

 #COULD USE A COMBINATION OF CNN AND RNN SO MODEL LEARNS MORE FORM SPATIAL AND TEMPORAL DEPENDENCIES    
    def forward(self, x):
        return self.layers(x) 



# Define General NV Class

class Navier_Stokes(nn.Module):
    def __init__(self, x_in, y_in, t_in, u_in, v_in):
        super().__init__()

        # define inputs (require_grd = True in case we operate on them)
        self.x = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(y_in, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(t_in, dtype=torch.float32, requires_grad=True)

        self.u = torch.tensor(u_in, dtype=torch.float32, requires_grad=True)
        self.v = torch.tensor(v_in, dtype=torch.float32, requires_grad=True)

        # define zeros tensor
        self.zeros = torch.zeros((self.x.shape[0], 1)) #N x 1

        # initialize loss and optimizer

        #perform hyperparameter search?

        # initialize network
        self.network = CustomNetwork()



        ### IMPORTANT ### when calculating the losses, it might be hard to determine which one is more important, 
        #it mght be goo to apply this algorithm: https://github.com/ranandalon/mtl, come from here https://arxiv.org/pdf/1705.07115.pdf (look in utls/loss_handler)

        #ideas
        #hyperparameter search (look at https://arxiv.org/pdf/2205.13748.pdf)
        #look for better architectures 
        #they've done the same as me! https://arxiv.org/pdf/2402.03153.pdf
        #should we add batch norm? (yes but if batch is enough big) stdi also gradient clipping


                 
   



print('ie')

