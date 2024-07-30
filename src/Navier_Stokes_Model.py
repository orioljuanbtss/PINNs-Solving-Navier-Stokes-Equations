
import os
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from dotenv import load_dotenv
from helpers import preprocess

# Load environment variables from .env file
load_dotenv()
project_path = os.environ.get('PROJECT_PATH')
data_path = os.environ.get('PROJECT_DATA_PATH')

# Load data
data = scipy.io.loadmat(Path(data_path) / Path('cylinder_wake.mat'))


### STEP 1: PreProcess data ###

processed_data = preprocess(data)


### STEP 2: Train model ###


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


                 
   

