import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps

# Try to solve the poisson equation:

'''
新的残差网络
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
    def __init__(self, in_dim=2, width=100, out_dim=1, depth=6, phi=PowerReLU):#phi=nn.Tanh
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


def get_interior_points(N=1024,d=2):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    x1 = torch.linspace(0, 1, 202)
    x2 = torch.linspace(0, 1, 202)
    X, Y = torch.meshgrid(x1, x2)  # 注意 PyTorch >=1.10 推荐加 indexing
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    return coords

def get_boundary_points(N=32):
    index = torch.rand(N, 1)
    index1 = torch.rand(N,1)
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)
    xb2 = torch.cat((index, torch.ones_like(index)), dim=1)
    xb3 = torch.cat((torch.zeros_like(index1), index1), dim=1)
    xb4 = torch.cat((torch.ones_like(index1), index1), dim=1)

    return xb1,xb2,xb3,xb4

def init_weights(m):
    """
    if isinstance(m, nn.Linear):
        # a: slope for ReLU — 设置为 PowerReLU 的导数近似（如 a=1）
        nn.init.kaiming_normal_(m.weight, a=3.0, nonlinearity='relu', mode='fan_in')
        # 手动缩放防爆（非常关键，建议 ×0.5 或更小）
        with torch.no_grad():
            m.weight *= 0.2
        nn.init.constant_(m.bias, 0.6)
    """
    if isinstance(m, nn.Linear):
        # a: slope for ReLU — 设置为 PowerReLU 的导数近似（如 a=1）
        nn.init.kaiming_normal_(m.weight, a=3.0, nonlinearity='relu', mode='fan_out')
        # 手动缩放防爆（非常关键，建议 ×0.5 或更小）
        with torch.no_grad():
            m.weight *= 0.1
        # # 2) 偏置处理
        if m.out_features == 1:
            # 最后一层 bias 设为 arctanh(0.6)
            b0 = math.atanh(0.02)  # ≈ 0.693147
            nn.init.constant_(m.bias, b0)



def main():

    epochs = 500

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Notice that the real code is "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
    # Although my computer supports cuda ,  its running speed is slower tahn 'cpu', ...........
    model = GlobalResNet(in_dim= 2, width=100, out_dim=1, depth=6, phi=PowerReLU).to(device) #phi=nn.tanh
    model.apply(init_weights)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)


    with torch.no_grad():
        coords = get_interior_points().to(device)  # [1002001, 2]
        # 模型预测
        pred = model(coords).cpu().numpy().reshape(202, 202)
    print(pred.mean())
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')#,vmin=0.29, vmax=0.31)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.title('initial')
    plt.show()


    # x = torch.cat((xr, xb), dim=0)

    # if 2 < m:
    #     y = torch.zeros(x.shape[0], m - 2)
    #     x = torch.cat((x, y), dim=1)
    # # print(x.shape)
    best_loss, best_epoch = 2000, 0
    losses = []
    losses_r1 = []
    losses_b1 = []
    losses_i1 = []
    for epoch in range(epochs+1):

        # generate the data set
        xr = get_interior_points()
        xb1, xb2, xb3, xb4 = get_boundary_points()
        xr = xr.to(device)
        xb1 = xb1.to(device)
        xb2 = xb2.to(device)
        xb3 = xb3.to(device)
        xb4 = xb4.to(device)
        xr.requires_grad_()
        output_r = model(xr)
        output_b1 = model(xb1)
        output_b2 = model(xb2)
        output_b3 = model(xb3)
        output_b4 = model(xb4)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        # print(grads)
        # loss_r = 0.5 * torch.sum(torch.pow(grads, 2),dim=1)- output_r
        loss_r = 0.5 * 0.04 ** 2 * torch.sum(torch.pow(grads, 2),dim=1) + 0.25 * ((output_r ** 2 -1) **2)
        loss_integ = torch.pow(output_r - torch.ones_like(output_r)*0.02,2)
        loss_r = torch.mean(loss_r)
        loss_b = torch.mean(torch.pow(output_b1 - output_b2,2)) + torch.mean(torch.pow(output_b3-output_b4,2))
        # loss_b = torch.mean(torch.abs(output_b1 - output_b2)) + torch.mean(torch.abs(output_b3-output_b4))   # 绝对值
        # loss = 4 * loss_r + 9 * 500 * loss_b
        loss_integ = torch.mean(loss_integ)
        loss = loss_r + 10 * loss_b + 10 * loss_integ

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        losses_r1.append(loss_r.item())
        losses_b1.append(loss_b.item())
        losses_i1.append(loss_integ.item())

        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(), 'loss_b:', loss_b.item(), 'loss_i:', loss_integ.item())
            if epoch > int(3 * epochs / 4):
                if torch.abs(loss) < best_loss:
                    best_loss = torch.abs(loss).item()
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'ritz_ch2.mdl')
    print('best epoch:', best_epoch, 'best loss:', best_loss)
    plt.figure()
    plt.plot(losses, color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_{total}$', font={'family': 'Arial', 'size': 14})
    plt.show()

    plt.figure()
    plt.plot((losses_r1), color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font={'family': 'Arial', 'size': 14})
    plt.savefig('loss_r.png')
    plt.show()

    plt.figure()
    plt.semilogy((losses_b1[1:]), color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_b$', font = {'family': 'Arial', 'size': 14})
    plt.savefig('loss_b.png')
    plt.show()

    plt.figure()
    plt.semilogy((losses_i1), color='black', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font = {'family': 'Arial', 'size': 14})
    plt.show()

    losses = np.array(losses)
    # 保存为.npy文件
    np.save('losst.npy', losses)

    losses_r1 = np.array(losses_r1)
    # 保存为.npy文件
    np.save('lossi.npy', losses_r1)

    losses_b1 = np.array(losses_b1)
    np.save('lossb.npy', losses_b1)

    losses_i1 = np.array(losses_i1)
    np.save('lossm.npy', losses_i1)

    # plot figure
    model.load_state_dict(torch.load('ritz_ch2.mdl'))
    print('load from ckpt!')
    with torch.no_grad():
        x1 = torch.linspace(0, 1, 202)
        x2 = torch.linspace(0, 1, 202)
        X, Y = torch.meshgrid(x1, x2)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        # if 2 < m:
        #     y = torch.zeros(Z.shape[0], m - 2)
        #     Z = torch.cat((Z, y), dim=1)
        Z = Z.to(device)
        pred = model(Z)
    plt.figure()
    pred = pred.cpu().numpy()
    pred = pred.reshape(202, 202)
    print(pred)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig('pred.png')
    plt.show()
    #设置范围

if __name__ == '__main__':
    main()

