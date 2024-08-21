import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


class CustomDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.float32)
        self.t = torch.tensor(data['t'], dtype=torch.float32)
        self.u = torch.tensor(data['u'], dtype=torch.float32)
        self.v = torch.tensor(data['v'], dtype=torch.float32)
        self.p = torch.tensor(data['p'], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.u[idx], self.v[idx], self.p[idx]
    

def preprocess_data(data, train_test_ratio = 0.9):
    """
    Preprocess the given data for training and testing.

    Parameters:
    data (dict): A dictionary containing the following keys:
        - 'U_star': Velocity components for each time step and mesh node (N x 2 x T)
        - 'p_star': Pressure for each time step and mesh node (N x T)
        - 't': Time steps (T x 1)
        - 'X_star': x/y coordinates of the mesh nodes (N x 2)

    Returns:
    tuple: Training and testing datasets containing:
        - x_train, y_train, t_train, u_train, v_train: Training data
        - x_test, y_test, t_test, u_test, v_test: Testing data
    """

    # Extract data
    U = data['U_star']  # N x 2 x T
    P = data['p_star']  # N x T
    t = data['t']  # T x 1
    X = data['X_star']  # N x 2

    N = X.shape[0] # number of nodes
    T = t.shape[0] # time steps

    # For testing we'll use the first time step
    x_test = X[:, 0:1]
    y_test = X[:, 1:2]
    p_test = P[:, 0:1]
    u_test = U[:, 0:1, 0]
    v_test = U[:, 1:2, 0] 
    t_test = np.ones((x_test.shape[0], x_test.shape[1]))

    # Compute indexes
    N_train = int(train_test_ratio * N * T)
    train_idx = np.random.choice(N * T, N_train, replace=False)

    # Rearrange Data
    XX = np.tile(X[:, 0:1], (1, T))  # N x T
    YY = np.tile(X[:, 1:2], (1, T))  # N x T
    TT = np.tile(t, (1, N)).T  # N x T

    UU = U[:, 0, :].flatten()  # NT
    VV = U[:, 1, :].flatten()  # NT
    PP = P.flatten()  # NT

    # Each node's data (x, y, t, u, v) needs to be associated with each time step. 
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU[:, None]  # NT x 1
    v = VV[:, None]  # NT x 1
    p = PP[:, None]  # NT x 1

    print(x.shape)
    train_data = {
        'x': x[train_idx],
        'y': y[train_idx],
        't': t[train_idx],
        'u': u[train_idx],
        'v': v[train_idx],
        'p': p[train_idx]
    }

    test_data = {
        'x': x_test,
        'y': y_test,
        't': t_test,
        'u': u_test[:, None],
        'v': v_test[:, None],
        'p': p_test
    }

    return {'train': train_data, 'test': test_data}


def get_train_data_loader(batch_size, num_workers, train_data):
    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader


