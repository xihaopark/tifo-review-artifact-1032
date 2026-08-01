import torch
import torch.nn as nn
import torch.optim as optim

# 计算统计量
def compute_statistics(X, U):
    return torch.matmul(X, U)

# 归一化
def normalize(X, Theta, U):
    return X - torch.matmul(Theta, U.T)

# 反归一化
def denormalize(Y_pred, Theta, V, H, L):
    return Y_pred + (Theta @ V.T) * torch.sqrt(torch.tensor(H / L))

# 损失函数
def loss_function(X, Y, U, V):
    Theta_x = compute_statistics(X, U)
    Theta_y = compute_statistics(Y, V)

    # 局部不变性损失
    L_loc_inv = torch.sum((Theta_x - Theta_y) ** 2)

    # 全局可变性损失
    L_glo_var = -torch.var(Theta_x) - torch.var(Theta_y)

    return L_loc_inv + L_glo_var

# 学习归一化和反归一化的函数
def learn_normalization(X, Y, num_features):
    L, H = X.shape[1], Y.shape[1]
    U = nn.Parameter(torch.randn(L, num_features))
    V = nn.Parameter(torch.randn(H, num_features))

    optimizer = optim.Adam([U, V], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = loss_function(X, Y, U, V)
        loss.backward()
        optimizer.step()

    return U.detach(), V.detach()