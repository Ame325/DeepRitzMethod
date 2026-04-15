import time

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
import random
from torch.optim import LBFGS
from numpy.fft import fft, ifft, fftfreq
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
    def __init__(self, width, phi=nn.Tanh): #phi=nn.Tanh
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            phi(),
            nn.Linear(width, width),
            phi(),
        )
    def forward(self, x):
        return x + 0.1 * self.block(x)

class ScaledBiasModel(nn.Module):
    def __init__(self, net, target_mean=0.02, init_scale=0.05):
        super().__init__()
        self.net = net
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.get_default_dtype()))
        self.bias = nn.Parameter(torch.tensor(float(target_mean), dtype=torch.get_default_dtype()))
    def forward(self, x):
        return self.bias + self.scale * self.net(x)

class GlobalResNet(nn.Module):
    def __init__(self, in_dim=1, width=100, out_dim=1, depth=6, phi=nn.Tanh):#phi=nn.Tanh
        super().__init__()
        self.input_layer = nn.Linear(in_dim, width)
        self.res_blocks   = nn.ModuleList([ResidualBlock(width, phi) for _ in range(depth)])
        self.output_layer = nn.Linear(width, out_dim)
        # self.act_out      = nn.Tanh()     # ← 最后一层添加 tanh

    def forward(self, x):
        out = self.input_layer(x)
        for blk in self.res_blocks:
            out = blk(out)
        return self.output_layer(out)
        # return self.act_out(self.output_layer(out))  # ← 包裹 tanh


def get_interior_points(N=101):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    x = torch.linspace(0, 1, N).unsqueeze(1)
    # x = x[torch.randperm(101)]
    return x

def get_boundary_points(N=256):
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
        if x.dtype != self.B.dtype:
            x = x.to(self.B.dtype)
        x_periodic = x % 1.0
        x_proj = 2 * torch.pi * x_periodic @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierSeriesFeature(nn.Module):
    """
    构造整数频率的 Fourier 特征（严格周期）。
    mode: "cartesian" / "separable" / "random_subset"
    in_dim: 输入维度（2）
    max_freq: 最大频率 K (使用 -K..K)
    random_M: 当 mode="random_subset" 时，从全组合中采样 M 个向量
    normalize: 是否对输出特征按 sqrt(feature_dim) 做缩放（避免数值过大）
    """
    def __init__(self, in_dim=2, max_freq=6, mode="separable", random_M=202, normalize=True, device='cpu'):
        super().__init__()
        assert mode in ("cartesian", "separable", "random_subset")
        self.in_dim = in_dim
        self.max_freq = max_freq
        self.mode = mode
        self.normalize = normalize
        self.device = device

        if mode == "separable":
            # 仅保留每维的 1D harmonics: k = 1..K (跳过 k=0 因为 cos(0)=1 可用 bias 表示)
            ks = list(range(1, max_freq+1))
            # 构建频率向量列表 Klist：例如 [(kx,0),(kx,0),...,(0,ky),...]
            Klist = []
            for dim in range(in_dim):
                for k in ks:
                    vec = [0]*in_dim
                    vec[dim] = k
                    Klist.append(vec)
            K = torch.tensor(Klist, dtype=torch.float32)  # [M, in_dim]
        else:
            # cartesian 或 random_subset
            ranges = [list(range(-max_freq, max_freq+1)) for _ in range(in_dim)]
            # 生成笛卡尔积（可能很大）
            from itertools import product
            all_K = [list(p) for p in product(*ranges)]  # length = (2K+1)^d
            if mode == "cartesian":
                K = torch.tensor(all_K, dtype=torch.float32)
            else:  # random_subset
                M = min(random_M, len(all_K))
                sel = random.sample(all_K, M)
                K = torch.tensor(sel, dtype=torch.float32)

        # 保存为 buffer（不训练）
        self.register_buffer('K', K)  # shape [M, in_dim]

    def forward(self, x):
        # x: [N, in_dim], 假定 x in [0,1]
        # proj = 2π * (x @ K^T)  -> [N, M]
        proj = 2 * math.pi * (x @ self.K.t().to(x.device))
        sinc = torch.sin(proj)
        cosc = torch.cos(proj)
        feats = torch.cat([sinc, cosc], dim=-1)  # [N, 2*M]
        if self.normalize:
            feats = feats / math.sqrt(feats.shape[-1])
        return feats

class L2Tracker:
    def __init__(self, Nx=200, device='cpu', dtype=None):
        self.Nx = Nx
        self.device = device
        self.dtype = dtype or torch.get_default_dtype()
        self.x = torch.linspace(0, 1, Nx, device=device, dtype=self.dtype)
        self.prev_phi = None
        self.L2_history = []

    @torch.no_grad()
    def update(self, phi):
        """
        phi: 当前网络预测值，shape (Nx*Nx,)
        """
        phi_1d = phi.view(self.Nx)#, self.Nx)

        if self.prev_phi is None:
            self.prev_phi = phi_1d.clone()
            return None
        else:
            diff = phi_1d - self.prev_phi
            I = torch.trapz(diff ** 2, x=self.x)
            # I = torch.trapz(I_x, x=self.x)
            L2 = torch.sqrt(I).item()
            self.prev_phi = phi_1d.clone()
            self.L2_history.append(L2)
            return L2

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
        nn.init.kaiming_normal_(m.weight, a=1.5, nonlinearity='relu', mode='fan_out')
        # 手动缩放防爆（非常关键，建议 ×0.5 或更小）
        with torch.no_grad():
            m.weight *= 0.3
        # # 2) 偏置处理
        if m.out_features == 1:
            # 最后一层 bias 设为 arctanh(0.6)
            # b0 = math.atanh(0.6)  # ≈ 0.693147
            nn.init.constant_(m.bias, 0.6)

def init_weights1(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))  # or 'linear' if linear out
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def lbfgs_refine(
    model,
    fourier,
    X_ref,
    Nx=200,
    target_mean=0.02,
    eps=0.01,
    lambda_param=0.0,
    mu=1.0,
    device='cuda',
    use_double=True,
    lbfgs_lr=1.0,
    max_iter=200,
    history_size=10,
    line_search_fn='strong_wolfe',
    verbose=True,
    lossi=None,
    lossm=None,
    tracker = L2Tracker()
):
    orig_device = next(model.parameters()).device
    orig_dtype = next(model.parameters()).dtype

    target_device = torch.device(device)
    model.to(target_device)
    dtype = torch.float64 if use_double else torch.float32
    if use_double:
        model.double()
    else:
        model.float()

    # X_ref -> leaf tensor with requires_grad
    X = X_ref.to(target_device).to(dtype).detach().clone().requires_grad_(True)

    lbfgs = LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn=line_search_fn
    )

    # tracker = L2Tracker(Nx=Nx, device=target_device, dtype=dtype)

    def compute_loss_on_X(x_in):
        x_mapped = fourier(x_in).double()
        out = model(x_mapped)
        phi = out.view(-1)

        grads = torch.autograd.grad(
            outputs=phi,
            inputs=x_in,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        term1 = 0.5 * (eps ** 2) * torch.sum(grads ** 2, dim=1)
        term2 = 0.25 * (phi ** 2 - 1) ** 2
        per_sample = term1 + term2

        loss_r = per_sample.mean()
        mean_r = phi.mean()
        constraint = mean_r - target_mean
        loss_penalty = 0.5 * mu * (constraint ** 2)
        loss = loss_r + loss_penalty + lambda_param * constraint
        lossi.append(loss_r.item())
        lossm.append(loss_penalty.item())

        return loss, loss_r.detach(), loss_penalty.detach(), mean_r.detach(), per_sample.detach(), phi.detach()

    def closure():
        lbfgs.zero_grad()

        # 保证 X 是 leaf tensor 且 requires_grad=True
        nonlocal X
        X = X.detach().clone().requires_grad_(True)

        loss, _, _, _, _, phi = compute_loss_on_X(X)
        loss.backward()

        L2_val = tracker.update(phi)
        if verbose and L2_val is not None:
            print(f"L2 between last two predictions: {L2_val:.5e}")
        return loss

    if verbose:
        print(f">>> Starting L-BFGS refine: use_double={use_double}, device={target_device}, max_iter={max_iter}")

    lbfgs.step(closure)


    loss_final, loss_r_final, loss_penalty_final, mean_final, _, _ = compute_loss_on_X(X)

    # restore model dtype/device
    model.to(orig_device)
    if orig_dtype == torch.float32:
        model.float()
    else:
        model.double()

    if verbose:
        print(f"LBFGS done. loss={loss_final.item():.6e}, loss_r={loss_r_final.item():.6e}, "
              f"loss_penalty={loss_penalty_final.item():.6e}, mean={mean_final.item():.6e}")

    # 绘制 L2 收敛
    if len(tracker.L2_history) > 0:
        plt.figure()
        plt.plot(range(len(tracker.L2_history)), tracker.L2_history)
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel(r'$\mathcal{force}$', font={'family': 'Arial', 'size': 14})
        # plt.title('L2 Convergence during L-BFGS')
        # plt.grid(True)
        plt.show()
    plt.figure()
    plt.plot(range(len(lossi)), lossi)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font={'family': 'Arial', 'size': 14})
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(range(len(lossm)),lossm)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font={'family': 'Arial', 'size': 14})
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    return loss_final.item(), loss_r_final.item(), loss_penalty_final.item(), mean_final.item(), tracker.L2_history

def main():
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ 
    X_sup = torch.linspace(0, 1, 101).unsqueeze(1)
    with open('y_sup.txt', 'r') as file:
        # 逐行读取并转换为浮点数列表
        data = []
        for line in file:
            # 按空格/逗号分割每行数据，并转换为float
            row = [float(x) for x in line.strip().split()]
            data.append(row)

    # 将数据转换为PyTorch张量
    Y_sup = torch.tensor(data)
    """
    scale_num = 3.0
    mapping_size = 16
    freq = 3
    ff = FourierFeature(1, half_mapping_size=mapping_size , scale=scale_num).to(device)
    # ff = FourierSeriesFeature(in_dim=1, max_freq=freq, mode="separable", normalize=True).to(device)
    # 1) 网络实例化：用 GlobalResNet 替换 drrnn 2 * ff.K.shape[0]
    # model = GlobalResNet(in_dim=2 * 2 * 16,  width=100, out_dim=1, depth=6, phi=nn.Tanh).to(device) #phi=nn.tanh
    net = GlobalResNet(in_dim= 2 * 2 * 16, width=80, out_dim=1, depth=4, phi=PowerReLU).to(device) #phi=nn.tanh
    model = ScaledBiasModel(net, target_mean=-0.2, init_scale=0.3).to(device)
    ler = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=ler)
    model.apply(init_weights)  #kaiming
    # fit_initial_output(model, X_sup, Y_sup, ff=ff, lr=5e-2, steps=1000, device=device)
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 101).unsqueeze(1).to(device)  # 1D 网格点
        x_plot_ff = ff(x_plot)  # 通过 Fourier 特征映射
        pred = model(x_plot_ff).cpu().numpy()  # 模型预测值
    plt.figure(figsize=(8, 4))
    plt.plot(x_plot.cpu().numpy(), pred, color='red', lw=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.ylim(-1.1, 1.1)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置 y 轴刻度间隔为 5
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.title('initial')
    plt.grid(True)
    plt.show()


    # === 进行监督预初始化 ===
    print(model)

    best_loss, best_epoch = 2000, 0
    losses = []
    losses_r1 = []
    losses_m1 = []
    un = []
    forces = []
    lr_flag = False
    # 超参数
    outer_iters = 5  # 外循环次数
    inner_iters = 100  # 每次内循环步数
    mu0 = 6.0  # 初始 penalty 系数
    rho = 1.2  # 每轮 mu 增长倍数
    mu_max = 10.0  # mu 上限
    target_mean = 0.6
    # 初始化拉格朗日乘子和 mu
    lambda_param = 0.0
    mu = mu0
    # 超参（可调）
    tau = 0.3  # relaxation（0.1~0.5）— 控制 lambda 更新幅度
    lambda_clip = 1e2  # 对 lambda 做上下界裁剪（根据你的loss scale调整）
    rho_increase = 1.2  # 当需要增加 mu 时用的较小倍率（比 1.2 保守）
    tol_constraint = 1e-3  # 目标容忍度
    improve_ratio = 1.02  # 认为 constraint 有“足够改善”的比例（2%）

    # 在外循环开始前（或第一次更新前）初始化一次 prev_constraint
    prev_constraint = float('inf')

    xn = get_interior_points(N=100).to(device)
    xn_f = ff(xn)

    tracker = L2Tracker(Nx=100, device=device, dtype=torch.float64)
    # 外循环：更新乘子 lambda
    for k in range(outer_iters):
        # 内循环：只用 penalty 项优化 u
        for t in range(inner_iters):
            model.train()
            optimizer.zero_grad()
            # if k>0 and t>10:
            #     for g in optimizer.param_groups:
            #         g['lr'] = ler
            # if lr_flag:
            #     for g in optimizer.param_groups:
            #         g['lr'] = ler * 0.2
            #         lr_flag = False
            # === 采样原始点 ===
            xr1 = get_interior_points().to(device)
            xr1.requires_grad_(True)
            x_r = ff(xr1)  # [N,2]

            # 网络输出
            out_r = model(x_r)

            # PDE 残差项
            grads = autograd.grad(outputs=out_r, inputs=xr1,
                                  grad_outputs=torch.ones_like(out_r),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            term1 = 0.5 * (0.04 ** 2) * torch.sum(grads ** 2, dim=1)
            term2 = 0.25 * (out_r ** 2 - 1) ** 2
            loss_r = (term1 + term2).mean()

            # penalty 项
            mean_r = out_r.mean()
            constraint = mean_r - target_mean
            loss_penalty = 0.5 * mu * constraint ** 2
            # === 梯度方差正则项（鼓励有起伏） ===
            # 总损失：

            loss = loss_r  + loss_penalty + lambda_param * constraint
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                phi = model(xn_f)
            tracker.update(phi)


            if k > int(3 * outer_iters / 4):
                if torch.abs(loss) < best_loss:
                    best_loss = torch.abs(loss).item()
                    # best_loss = torch.abs(min_loss-torch.ones_like(min_loss)*0.0751).item() + min_loss.item()
                    best_epoch = t + k * inner_iters
                    torch.save(model.state_dict(), 'ritz_1dim.mdl')
            if k == t == 0:
               print(out_r.mean())
            losses.append(loss.item())
            losses_r1.append(loss_r.item())
            losses_m1.append(loss_penalty.item())
            if t % 100 == 0:
                print(f"epoch:{t + k * inner_iters}, loss_r:{loss_r.item()}, loss_penalty:{loss_penalty.item()}")
        # 内循环结束，计算当前 mean
        with torch.no_grad():
            mean_hat = model(xn_f).mean().item()


        # # 外循环更新 lambda 和 mu
        # # 在外循环中，当你已计算 mean_hat:
        # constraint = mean_hat - target_mean
        # abs_constraint = abs(constraint)
        #
        # # decide whether constraint improved compared to previous recorded
        # improved = (prev_constraint / (abs_constraint + 1e-12)) > improve_ratio
        # # 只在满足一定条件时更新 lambda，并用 relaxation + clipping
        # # 例如：若 inner loop 足够做了，且 constraint 不是非常小
        # if abs_constraint > tol_constraint:
        #     # relaxation 更新，避免一次性过大跳动
        #     lambda_param = lambda_param + tau * mu * constraint
        #     # 裁剪 lambda 防止发散
        #     lambda_param = max(min(lambda_param, lambda_clip), -lambda_clip)
        #     lr_flag = True
        #
        # # 只在 constraint 没有明显改善时温和增大 mu
        # if not improved and (abs_constraint > tol_constraint):
        #     mu = min(mu * rho_increase, mu_max)
        # # 如果有明显改善，可以不增大 mu，甚至按需要轻微减小或固定
        # # else:
        # #     mu = max(mu / 1.02, mu0)  # 可选：小幅减小或保持
        #
        # # 记录当前 constraint 以便下一次比较
        # prev_constraint = max(abs_constraint, 1e-12)
        lambda_param += mu * (mean_hat - target_mean)
        mu = min(mu * rho, mu_max)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        print(f"mean={mean_hat:.4f}, constraint={constraint:.5f}, "
              f"lambda={lambda_param:.4f}, mu={mu:.2f}, grad_norm={grad_norm.item()}")

        #print(f"Outer {k}: mean={mean_hat:.4f}, lambda={lambda_param:.4f}, mu={mu:.2f}")
        model.eval()  # 切换为评估模式
        with torch.no_grad():
            x_plot = torch.linspace(0, 1, 101).unsqueeze(1).to(device)  # 1D 网格点
            x_plot_ff = ff(x_plot)  # 通过 Fourier 特征映射
            pred = model(x_plot_ff).cpu().numpy()  # 模型预测值
        plt.figure(figsize=(8, 4))
        plt.plot(x_plot.cpu().numpy(), pred, color='red', lw=2)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.ylim(-1.1, 1.1)
        # 获取当前坐标轴
        ax = plt.gca()
        # 设置 y 轴刻度间隔为 5
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        plt.title(f'epoch:{(k+1) * inner_iters} Output')
        plt.grid(True)
        plt.show()

    loss_final, loss_r_final, loss_penalty_final, mean_final, L2_history = lbfgs_refine(
        model=model,
        fourier=ff,
        X_ref=xn.double(),
        Nx=100,
        target_mean=0.6,  # 你之前的目标均值
        eps=0.04,
        lambda_param=lambda_param,  # 可根据你的增广拉格朗日调整
        mu=mu,  # 增广项系数
        device='cuda',
        use_double=True,
        lbfgs_lr=0.8,
        max_iter=100,
        history_size=10,
        line_search_fn='strong_wolfe',
        verbose=True,
        lossi=losses_r1,
        lossm=losses_m1,
        tracker = tracker
    )

    torch.save(model.state_dict(), 'ritz_lgb1d.mdl')
    print("Final L2 history length:", len(L2_history))

    print('best epoch:', best_epoch, 'best loss:', best_loss)
    plt.figure()
    plt.plot(losses, color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_t$', font={'family': 'Arial', 'size': 14})

    plt.show()
    plt.plot((losses_r1), color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font={'family': 'Arial', 'size': 14})

    plt.savefig('loss_r.png')
    plt.show()
    plt.semilogy((losses_m1), color='green', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font={'family': 'Arial', 'size': 14})
    plt.show()

    print('force:', tracker.L2_history)
    plt.semilogy(tracker.L2_history, color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{force}$', font = {'family': 'Arial', 'size': 14})
    plt.show()

    losses_r1 = np.array(losses_r1)
    # 保存为.npy文件
    np.save('lossi.npy', losses_r1)

    losses_m1 = np.array(losses_m1)
    # 保存为.npy文件
    np.save('lossm.npy', losses_m1)

    forces = np.array(tracker.L2_history)
    # 保存为.npy文件
    np.save('force.npy', forces)


    # plot figure
    model.load_state_dict(torch.load('ritz_lgb1d.mdl'))
    model.eval()
    print('load from ckpt!')
    with torch.no_grad():
        x1 = torch.linspace(0, 1, 101).unsqueeze(1).to(device)
        x_ff = ff(x1)
        # 模型预测
        pred = model(x_ff).cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x1.cpu().numpy(), pred, color='red', lw=2)
    # plt.title('pred')
    plt.ylim(-1.1, 1.1)
    # 获取当前坐标轴
    ax = plt.gca()
    # 设置 y 轴刻度间隔为 5
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.xlabel('x')
    plt.ylabel('u(x)')
    # plt.grid(True)
    plt.show()
    #设置范围

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"代码执行时间：{execution_time//60} 分,{execution_time%60:.1f}秒")
    print(f"代码执行时间：{execution_time:.2f}秒")
