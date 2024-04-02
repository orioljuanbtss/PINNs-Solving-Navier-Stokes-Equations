
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from fenics import *


model_path = Path.cwd() / Path('app') /  Path('models')
data_path =  Path.cwd() / Path('app') / Path('data')

data = scipy.io.loadmat(data_path / Path('cylinder_wake.mat'))

print('hey')