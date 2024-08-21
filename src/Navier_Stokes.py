
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad


class CNN(nn.Module):
    def __init__(self, input_channels, num_filters, filter_size, pool_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, filter_size, padding=1)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, filter_size, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output two values: psi and p
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x.to(self.device)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
    

class Navier_Stokes(nn.Module):
    def __init__(self, cnn, rnn, nu, optimizer):
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.net = nn.Sequential(cnn, rnn)
        self.nu = nu
        self.optimizer = optimizer
        self.mse = nn.MSELoss()
        self.iter = 0

    def forward(self, x, y, t):
        spatial_input = torch.stack((x, y), dim=1).unsqueeze(1)  # Shape: (NT, 1, 2)
        spatial_features = self.cnn(spatial_input)  # Shape: (NT, num_filters*2, 1, 1)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        combined_input = torch.cat((spatial_features, t), dim=1)  # Shape: (NT, num_filters*2 + 1)
        combined_input = combined_input.view(-1, t.size(0) // x.size(0), combined_input.size(1))  # Shape: (N, T, input_size)
        output = self.rnn(combined_input)
        psi, p = output[:, 0], output[:, 1]
        return psi, p

    def function(self, x, y, t):
        psi, p = self.forward(x, y, t)
        u = grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        return u, v, p, f, g

    def closure(self, batch):
        x, y, t, u, v, p = batch
        self.optimizer.zero_grad()
        u_pred, v_pred, p_pred, f_pred, g_pred = self.function(x, y, t)
        u_loss = self.mse(u_pred, u)
        v_loss = self.mse(v_pred, v)
        f_loss = self.mse(f_pred, torch.zeros_like(f_pred))
        g_loss = self.mse(g_pred, torch.zeros_like(g_pred))
        loss = u_loss + v_loss + f_loss + g_loss
        loss.backward()
        return loss

    def train_model(self, train_loader, num_iterations):
        self.net.train()
        for _ in range(num_iterations):
            for batch in train_loader:
                self.optimizer.step(lambda: self.closure(batch))

