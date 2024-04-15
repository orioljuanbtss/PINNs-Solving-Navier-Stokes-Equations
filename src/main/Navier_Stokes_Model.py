
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


print('iep')
### STEP 1: PreProcessing ###



#class Navier_Stokes()