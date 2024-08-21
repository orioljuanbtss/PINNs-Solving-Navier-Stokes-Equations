
import os
import torch
from pathlib import Path
import torch.nn as nn
import multiprocessing as mp
import scipy.io
from dotenv import load_dotenv
from helpers import preprocess_data, get_train_data_loader
from Navier_Stokes import Navier_Stokes, CNN, RNN

# Load environment variables from .env file
load_dotenv()
project_path = os.environ.get('PROJECT_PATH')
data_path = os.environ.get('PROJECT_DATA_PATH')

# Load data
data = scipy.io.loadmat(Path(data_path) / Path('cylinder_wake.mat'))
num_workers = mp.cpu_count()


### STEP 1: PreProcess data ###
dataset = preprocess_data(data)
dataset_train = dataset['train']
dataset_test = dataset['test']

#train_loader = get_train_data_loader(32, num_workers, dataset_train)

print(dataset_train)


# # Example usage
# input_channels = 1
# num_filters = 16
# filter_size = 3
# pool_size = 2
# cnn = CNN(input_channels, num_filters, filter_size, pool_size)

# input_size = num_filters * 2  # Adjust according to the output size of CNN
# hidden_size = 50
# num_layers = 2
# rnn = RNN(input_size, hidden_size, num_layers)

# nu = 0.01  # Example kinematic viscosity
# optimizer = torch.optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=1e-3)

# pinn = Navier_Stokes(cnn, rnn, nu, optimizer)

# # Train the model
# pinn.train_model(train_loader, num_iterations=1000)

# # Save the trained model
# torch.save(pinn.state_dict(), 'model.pt')

# # Load the trained model
# pinn.load_state_dict(torch.load('model.pt'))
# pinn.eval()

# # Perform inference
# u_out, v_out, p_out, f_out, g_out = pinn.function(dataset_test['x'], dataset_test['y'], dataset_test['t'])
# print("u_out:", u_out)
# print("v_out:", v_out)
# print("p_out:", p_out)
# print("f_out:", f_out)
# print("g_out:", g_out)
   





# #         ### IMPORTANT ### when calculating the losses, it might be hard to determine which one is more important, 
# #         #it mght be goo to apply this algorithm: https://github.com/ranandalon/mtl, come from here https://arxiv.org/pdf/1705.07115.pdf (look in utls/loss_handler)

# #         #ideas
# #         #hyperparameter search (look at https://arxiv.org/pdf/2205.13748.pdf)
# #         #look for better architectures 
# #         #they've done the same as me! https://arxiv.org/pdf/2402.03153.pdf
# #         #should we add batch norm? (yes but if batch is enough big) stdi also gradient clipping


                 
   

