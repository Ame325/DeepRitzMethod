# import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
torch.pi = math.pi
# Try to solve the equation:

'''

'''
class PowerReLU(nn.Module):
    def __init__(self, inplace=False, power=1.5):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power
    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)

class ResidualBlock(nn.Module):
    def __init__(self, width, phi=PowerReLU): #phi=nn.Tanh
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            phi(),
            nn.Linear(width, width),
            phi(),
        )
    def forward(self, x):
        return x + self.block(x)

class GlobalResNet(nn.Module):
    def __init__(self, in_dim=1, width=100, out_dim=1, depth=6, phi=PowerReLU):#phi=nn.Tanh
        super().__init__()
        self.input_layer = nn.Linear(in_dim, width)
        self.res_blocks   = nn.ModuleList([ResidualBlock(width, phi) for _ in range(depth)])
        self.output_layer = nn.Linear(width, out_dim)
        self.act_out      = nn.Tanh()     # ← 最后一层添加 tanh

    def forward(self, x):
        out = self.input_layer(x)
        for blk in self.res_blocks:
            out = blk(out)
        # return self.output_layer(out)
        return self.act_out(self.output_layer(out))  # ← 包裹 tanh


def get_interior_points(N=1024,d=1):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    x = torch.linspace(0, 1, 101).unsqueeze(1)
    # x = x[torch.randperm(101)]
    return x

def get_boundary_points(N=32):
    xb1 = torch.ones(N, 1)
    xb2 = torch.zeros(N, 1)
    return xb1,xb2


class FourierFeature(nn.Module):
    # def __init__(self, in_features, mapping_size=64, scale=10.0):
    #     super().__init__()
    #     self.B = nn.Parameter(scale * torch.randn((in_features, mapping_size)), requires_grad=False)
    def __init__(self, in_features=1, half_mapping_size=32, scale=10.0):
        super().__init__()
        B_half = scale * torch.randn((in_features, half_mapping_size))
        B = torch.cat([B_half, -B_half], dim=1)  # 构造正负对称频率
        self.register_buffer('B', B)  # [in_features, mapping_size]
    def forward(self, x):
        x_periodic = x % 1.0
        x_proj = 2 * torch.pi * x_periodic @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def fit_initial_output(model, X_sup, Y_sup, ff=None, lr=1e-2, steps=1000, device='cuda'):
    """
    用于预拟合网络使其初始输出接近 Y_sup（拟合监督解）
    """
    model.eval()
    model.to(device)

    # 暂时只训练 output 层
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.output_layer.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.output_layer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_input = X_sup.to(device)
    if ff:
        X_input = ff(X_input)

    Y_target = Y_sup.to(device)

    for step in range(steps):
        pred = model(X_input)
        loss = loss_fn(pred, Y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"[InitFit] step {step}, loss = {loss.item():.6f}")

    # 恢复参数为可训练
    for param in model.parameters():
        param.requires_grad = True


def init_weights(m):
    if isinstance(m, nn.Linear):
        # a: slope for ReLU — 设置为 PowerReLU 的导数近似（如 a=1）
        nn.init.kaiming_normal_(m.weight, a=3.0, nonlinearity='relu', mode='fan_out')
        # 手动缩放防爆（非常关键，建议 ×0.5 或更小）
        with torch.no_grad():
            m.weight *= 0.3
        # # 2) 偏置处理
        if m.out_features == 1:
            # 最后一层 bias 设为 arctanh(0.6)
            b0 = math.atanh(0.6)  # ≈ 0.693147
            nn.init.constant_(m.bias, b0)


def main():
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) 网络实例化：用 GlobalResNet 替换 drrnn
    model = GlobalResNet(in_dim=1, width=100, out_dim=1, depth=6, phi=PowerReLU).to(device) #phi=nn.tanh
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.apply(init_weights)  #kaiming
    # fit_initial_output(model, X_sup, Y_sup, ff=ff, lr=5e-2, steps=1000, device=device)
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 101).unsqueeze(1).to(device)  # 1D 网格点
        pred = model(x_plot).cpu().numpy()  # 模型预测值
    plt.figure(figsize=(8, 4))
    plt.plot(x_plot.cpu().numpy(), pred, color='red', lw=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    # plt.ylim(-1.1, 1.1)
    # 获取当前坐标轴
    plt.title('initial')
    # plt.grid(True)
    plt.show()


    # === 进行监督预初始化 ===
    print(model)
    epochs = 500
    best_loss, best_epoch = 2000, 0
    losses = []
    losses_r1 = []
    losses_b1 = []
    losses_m1 = []

    alpha = 5.0 #边界损失权重
    target_mean = 0.6
    # 初始化拉格朗日乘子和 mu
    lambda_param = 0.0
    for epoch in range(epochs+1):
            model.train()
            optimizer.zero_grad()

            # === 采样原始点 ===
            xr1 = get_interior_points().to(device)
            xr1.requires_grad_(True)

            # 网络输出
            out_r = model(xr1)

            xb1, xb2 = get_boundary_points()
            output_b1 = model(xb1.to(device))
            output_b2 = model(xb2.to(device))

            # PDE 残差项
            grads = autograd.grad(outputs=out_r, inputs=xr1,
                                  grad_outputs=torch.ones_like(out_r),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss_r = 0.5 * 0.04 ** 2 * torch.sum(torch.pow(grads, 2), dim=1) + 0.25 * ((out_r ** 2 - 1) ** 2)
            loss_r = torch.mean(loss_r)
            #boundary
            # loss_b = torch.mean(torch.pow(output_b1 - output_b2,2))
            loss_b = torch.mean(torch.abs(output_b1 - output_b2))
            # penalty 项
            mean_r = out_r.mean()
            constraint = mean_r - target_mean
            # === 梯度方差正则项（鼓励有起伏） ===
            # 总损失：
            loss_integ = torch.pow(out_r - torch.ones_like(out_r) * 0.6, 2)
            loss_integ = torch.mean(loss_integ)
            loss = loss_r  + 500 * loss_integ + 1 * loss_b
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            losses_r1.append(loss_r.item())
            losses_b1.append(loss_b.item())
            losses_m1.append(loss_integ.item())
            if epoch % 100 == 0:
                print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(), 'loss_b:', loss_b.item(),
                      'loss_i:', loss_integ.item())
                print(mean_r.item())
                if epoch > int(3 * epochs / 4):
                    if torch.abs(loss) < best_loss:
                        best_loss = torch.abs(loss).item()
                        best_epoch = epoch
                        torch.save(model.state_dict(), 'ritz_ch1.mdl')
    print('best epoch:', best_epoch, 'best loss:', best_loss)
    plt.figure()
    plt.plot(losses, color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_{total}$', font={'family': 'Arial', 'size': 14})
    plt.show()

    plt.figure()
    plt.plot(losses_r1, color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font={'family': 'Arial', 'size': 14})
    plt.savefig('loss_r.png')
    plt.show()

    plt.figure()
    plt.semilogy(losses_b1[1:], color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_b$', font={'family': 'Arial', 'size': 14})
    plt.show()


    plt.semilogy(losses_m1, color='black', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font={'family': 'Arial', 'size': 14})
    plt.show()

    # plot figure
    model.load_state_dict(torch.load('ritz_ch1.mdl'))
    model.eval()
    print('load from ckpt!')
    with torch.no_grad():
        x1 = torch.linspace(0, 1, 101).unsqueeze(1).to(device)
        # 模型预测
        pred = model(x1).cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x_plot.cpu().numpy(), pred, color='red', lw=2)
    plt.title('pred')
    plt.ylim(-1.1, 1.1)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置 y 轴刻度间隔为 5
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.show()
    #设置范围

if __name__ == '__main__':
    main()
