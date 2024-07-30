import numpy as np

def preprocess(data, train_test_ratio = 0.9):
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

    # Perform 90-10 sampling 
    N_train = int(train_test_ratio * N * T)

    # Create train/test indices
    total_indices = np.arange(N * T)
    train_idx = np.random.choice(total_indices, N_train, replace=False)
    test_idx = np.setdiff1d(total_indices, train_idx) 

    # Rearrange Data
    XX = np.tile(X[:, 0], T)  # NT 
    YY = np.tile(X[:, 1], T)  # NT
    TT = np.repeat(t.flatten(), N)  # NT

    UU = U[:, 0, :].flatten()  # NT
    VV = U[:, 1, :].flatten()  # NT
    PP = P.flatten()  # NT

    # Each node's data (x, y, t, u, v) needs to be associated with each time step. 
    x = XX[:, None]  # NT x 1
    y = YY[:, None]  # NT x 1
    t = TT[:, None]  # NT x 1
    u = UU[:, None]  # NT x 1
    v = VV[:, None]  # NT x 1
    p = PP[:, None]  # NT x 1

    train_data = {
        'x': x[train_idx],
        'y': y[train_idx],
        't': t[train_idx],
        'u': u[train_idx],
        'v': v[train_idx]
    }

    test_data = {
        'x': x[test_idx],
        'y': y[test_idx],
        't': t[test_idx],
        'u': u[test_idx],
        'v': v[test_idx]
    }


    return {'train': train_data, 'test': test_data}
