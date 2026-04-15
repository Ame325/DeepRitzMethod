import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import random
import time
torch.pi = math.pi
from typing import Tuple
from torch.quasirandom import SobolEngine
from torch.optim import LBFGS
# Try to solve the equation:

'''
新的残差网络
'''
DTYPE = torch.float64
class PowerReLU(nn.Module):
    def __init__(self, inplace=False, power=1.5):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power
    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)

class Sine(nn.Module):
    def __init__(self, omega_0=20.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

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
        return x + 1.0 * self.block(x)

class ScaledBiasModel(nn.Module):
    def __init__(self, net, target_mean=0.02, init_scale=0.05):
        super().__init__()
        self.net = net
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.get_default_dtype()))
        self.bias = nn.Parameter(torch.tensor(float(target_mean), dtype=torch.get_default_dtype()))
    def forward(self, x):
        return self.bias + self.scale * self.net(x)


class GlobalResNet(nn.Module):
    def __init__(self, in_dim=2, width=100, out_dim=1, depth=6, phi=nn.Tanh):#phi=nn.Tanh
        super().__init__()
        self.input_layer = nn.Linear(in_dim, width)
        self.res_blocks   = nn.ModuleList([ResidualBlock(width, phi) for _ in range(depth)])
        self.output_layer = nn.Linear(width, out_dim)
        self.act_out      = nn.Identity()     # ← 最后一层添加 tanh
        # self.act_out = nn.Tanh()  # ← 最后一层添加 tanh

    def forward(self, x):
        out = self.input_layer(x)
        for blk in self.res_blocks:
            out = blk(out)
        return self.output_layer(out)
        # return self.act_out(self.output_layer(out))  # ← 包裹 tanh

# --- 1. 定义 FourierFeature 模块 ---

class FourierSeriesFeature(nn.Module):
    """
    构造整数频率的 Fourier 特征（严格周期）。
    mode: "cartesian" / "separable" / "random_subset"
    in_dim: 输入维度（2）
    max_freq: 最大频率 K (使用 -K..K)
    random_M: 当 mode="random_subset" 时，从全组合中采样 M 个向量
    normalize: 是否对输出特征按 sqrt(feature_dim) 做缩放（避免数值过大）
    """
    def __init__(self, in_dim=2, max_freq=6, mode="separable",
                 random_M=202, normalize=True, device='cpu', freq_scale=1.0, dtype=torch.float64):
        super().__init__()
        assert mode in ("cartesian", "separable", "random_subset")
        self.in_dim = in_dim
        self.max_freq = max_freq
        self.mode = mode
        self.normalize = normalize
        self.device = device
        self.freq_scale = float(freq_scale)
        self.dtype = dtype


        if mode == "separable":
            # 仅保留每维的 1D harmonics: k = 1..K (跳过 k=0 因为 cos(0)=1 可用 bias 表示)
            ks = list(range(0, max_freq+1))
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
        # ensure x dtype matches K
        if x.dtype != self.K.dtype:
            x = x.to(self.K.dtype)
        proj = 2 * math.pi * self.freq_scale * (x @ self.K.t().to(x.device))
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
        phi_2d = phi.view(self.Nx, self.Nx)

        if self.prev_phi is None:
            self.prev_phi = phi_2d.clone()
            return None
        else:
            diff = phi_2d - self.prev_phi
            I_x = torch.trapz(diff ** 2, x=self.x, dim=1)
            I = torch.trapz(I_x, x=self.x)
            L2 = torch.sqrt(I).item()
            self.prev_phi = phi_2d.clone()
            self.L2_history.append(L2)
            return L2

def get_interior_points(N=202):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    x1 = torch.linspace(0, 1, N)
    x2 = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x1, x2)  # 注意 PyTorch >=1.10 推荐加 indexing
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    return coords

def sample_sobol_interior(N, device):
    """
    返回 N 个 Sobol 低差异采样点，范围 [0,1]^2
    """
    eng = torch.quasirandom.SobolEngine(2, scramble=True)
    pts = eng.draw(N).to(device)      # shape [N,2]
    pts = pts.to(torch.get_default_dtype())
    return pts

def adaptive_sobol_2d_deep_ritz(model, X_raw, fourier, device, N_add=1000, top_ratio=0.2, region_size=0.05, eps=0.01):
    """
    二维 Sobol 自适应加密采样，结合 Deep Ritz PDE 残差
    Args:
        model: 神经网络
        X_raw: 原始训练点, shape [N, 2], requires_grad=True
        fourier: Fourier 映射函数
        N_add: 新增采样点总数
        top_ratio: 取高残差前 top_ratio 区域加密采样
        region_size: 每个高残差点的采样区域大小
        eps: PDE 参数
    Returns:
        X_new: 新增采样点, shape [N_add, 2]
    """

    # ---------- 定义 PDE 残差函数 ----------
    def pde_residual_fn(model, x_r_raw, fourier, eps=eps):
        x_r_raw.requires_grad_(True)
        # Fourier 映射
        x_r = fourier(x_r_raw)  # [N, 2*mapping_size]

        # 网络输出
        out_r = model(x_r)      # [N,1] 或 [N,]
        phi = out_r.view(-1)    # [N]

        # 一阶导
        grads = torch.autograd.grad(outputs=phi,
                                    inputs=x_r_raw,
                                    grad_outputs=torch.ones_like(phi),
                                    create_graph=False)[0]  # [N, 2]

        # PDE 残差项（每点）
        term1 = 0.5 * (eps ** 2) * torch.sum(grads ** 2, dim=1)      # [N]
        term2 = 0.25 * (phi ** 2 - 1) ** 2                            # [N]

        r = term1 + term2  # [N]
        return r

    # ---------- 计算残差 ----------
    model.eval()

    r = pde_residual_fn(model, X_raw, fourier)  # [N]
    r_abs = r.abs()

    # 选取高残差点索引
    N_high = max(1, int(len(X_raw) * top_ratio))
    _, idx = torch.topk(r_abs, N_high)
    X_high = X_raw[idx]

    # 每个高残差点生成 Sobol 网格点数
    points_per_region = (N_add // N_high) + 1
    X_new_list = []

    for x in X_high:
        sobol = SobolEngine(dimension=2, scramble=True)
        pts = sobol.draw(points_per_region).to(device)
        # 缩放到局部 region_size 并平移到高残差点附近
        pts = (pts - 0.5) * region_size + x
        pts = torch.clamp(pts, 0.0, 1.0)  # 保证在域 [0,1]
        X_new_list.append(pts)

    X_new = torch.cat(X_new_list, dim=0)[:N_add]

    return X_new.detach()



def fit_initial_output_2d_ls_then_finetune(model,
                                           X_sup,
                                           Y_sup,
                                           ff=None,
                                           device=None,
                                           ridge=1e-8,
                                           finetune=True,
                                           ft_lr=1e-4,
                                           ft_steps=2000,
                                           ft_print_every=200):
    """
    用法：
        model:      GlobalResNet 实例 (包含 input_layer, res_blocks, output_layer)
        X_sup:      torch tensor, shape [N,2] 坐标网格 (values in [0,1])
        Y_sup:      torch tensor, shape [H, W] 或 [N,] / [N,1] 的真实场
        ff:         FourierFeature 或其他特征映射函数（callable），若无则为 None
        device:     torch.device，若 None 自动选用 cuda/CPU
        ridge:      ridge 正则系数，用于稳定求解
        finetune:   是否在写回线性解后做全参微调
        ft_lr:      全参微调学习率
        ft_steps:   全参微调步数
        ft_print_every: 微调期间打印间隔
    返回：
        dict 包含 'mse_ls' (线性最小二乘 mse), 'mse_after_write' (写回后立即评估),
        'mse_final' (微调后 mse 或 None 如果未微调)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # 预处理 X, Y
    X_t = X_sup.to(device).float()         # [N, 2]
    if ff is not None:
        X_in = ff(X_t)                     # [N, D_in]
    else:
        X_in = X_t

    # 目标 flatten -> [N,1]
    if Y_sup.dim() == 2:
        Y_t = Y_sup.reshape(-1, 1).to(device).float()
    elif Y_sup.dim() == 1:
        Y_t = Y_sup.unsqueeze(1).to(device).float()
    else:
        Y_t = Y_sup.reshape(-1, 1).to(device).float()

    N = X_in.shape[0]
    print(f"[LS-init] samples: {N}, feature_in_dim: {X_in.shape[1]}")

    # 1) 计算隐藏表征 H = model_without_output(X_in)
    with torch.no_grad():
        h = model.net.input_layer(X_in)
        for blk in model.net.res_blocks:
            h = blk(h)
        H = h.detach()    # [N, D_h]
    D = H.shape[1]
    print(f"[LS-init] hidden dim D = {D}")

    # 2) 在 H 上求 ridge 最小二乘（用 double 以提升数值稳定性）
    H_np = H.cpu().numpy()        # [N, D]
    y_np = Y_t.cpu().numpy()      # [N, 1]
    ones = np.ones((N, 1), dtype=H_np.dtype)

    H_aug = np.concatenate([H_np, ones], axis=1)   # [N, D+1]
    A = H_aug.T.dot(H_aug)                         # [D+1, D+1]
    reg = ridge * np.eye(D + 1, dtype=A.dtype)
    A_reg = A + reg
    rhs = H_aug.T.dot(y_np)                        # [D+1, 1]

    # solve
    coef = np.linalg.solve(A_reg, rhs)             # [D+1, 1]
    w_ls = coef[:-1].reshape(1, -1)                # [1, D]
    b_ls = float(coef[-1])

    # compute LS mse (cpu)
    pred_ls = H_aug.dot(coef)
    mse_ls = float(np.mean((pred_ls - y_np)**2))
    print(f"[LS-init] ridge={ridge:.0e}, LS MSE = {mse_ls:.6e}")

    # 3) 写回 output_layer (注意 shape 匹配)
    out_w_shape = model.net.output_layer.weight.shape   # (1, D_model)
    if out_w_shape[1] != D:
        raise RuntimeError(f"output_layer expects input dim {out_w_shape[1]} but hidden dim is {D}. "
                           "请确认 model.input_layer 与 ff 的输出维度一致。")

    with torch.no_grad():
        model.net.output_layer.weight.copy_(torch.from_numpy(w_ls).float().to(device))
        model.net.output_layer.bias.fill_(b_ls)

    # 评估写回后的 mse (full model forward)
    model.eval()
    with torch.no_grad():
        pred_after = model(X_in).detach().cpu().numpy()    # [N,1]
        mse_after = float(np.mean((pred_after - y_np)**2))
    print(f"[LS-init] MSE after writing weights: {mse_after:.6e} (should ≈ LS MSE)")

    result = {'mse_ls': mse_ls, 'mse_after_write': mse_after, 'mse_final': None}

    # 4) 可选：全参微调（小 lr）
    if finetune:
        print("[Finetune] start full-parameter fine-tuning ...")
        # 解冻全部参数
        for p in model.net.parameters():
            p.requires_grad = True
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=ft_lr, weight_decay=1e-6)
        loss_fn = nn.MSELoss()

        X_in_train = X_in  # full-batch
        Y_train = Y_t

        for it in range(ft_steps):
            optimizer.zero_grad()
            pred = model(X_in_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            if (it % ft_print_every == 0) or (it == ft_steps - 1):
                print(f"[Finetune] it {it:5d}/{ft_steps}, loss = {loss.item():.6e}")
        # 结束微调，评估
        model.eval()
        with torch.no_grad():
            pred_final = model(X_in).detach().cpu().numpy()
            mse_final = float(np.mean((pred_final - y_np)**2))
        print(f"[Finetune] final MSE = {mse_final:.6e}")
        result['mse_final'] = mse_final

    return result

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
        nn.init.kaiming_normal_(m.weight, a=1.5, nonlinearity='relu', mode='fan_out')
        # 手动缩放防爆（非常关键，建议 ×0.5 或更小）
        with torch.no_grad():
            m.weight *= 0.3
        # # 2) 偏置处理
        if m.out_features == 1:
            # 最后一层 bias 设为 arctanh(0.6)
            # b0 = math.atanh(0.02)  # ≈ 0.693147
            nn.init.constant_(m.bias, 0.02)
            # nn.init.constant_(m.bias, b0)

def init_weights1(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))  # or 'linear' if linear out
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def sine_init(m, is_first=False, omega_0=30.0):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            if is_first:
                m.weight.uniform_(-1/m.in_features, 1/m.in_features)
            else:
                bound = math.sqrt(6 / m.in_features) / omega_0
                m.weight.uniform_(-bound, bound)
            if m.bias is not None:
                m.bias.fill_(0)



def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights on [0,1] (mapped from [-1,1]). Numpy arrays."""
    nodes, weights = np.polynomial.legendre.leggauss(n)  # nodes in [-1,1]
    nodes01 = 0.5 * (nodes + 1.0)  # map to [0,1]
    weights01 = 0.5 * weights
    return nodes01, weights01

def gauss_legendre_2d(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return coords (n*n,2) and weights (n*n) as torch tensors (dtype=DTYPE) on device.
    coords ordered row-major (y outer, x inner) so that phi.view(n,n) aligns with weights.reshape(n,n).
    """
    nodes, weights = gauss_legendre_1d(n)
    XX, YY = np.meshgrid(nodes, nodes, indexing='xy')
    WW = np.outer(weights, weights)  # shape (n,n)
    coords = np.stack([XX.flatten(), YY.flatten()], axis=-1)  # (n*n,2)
    coords_t = torch.tensor(coords, dtype=torch.float64, device=device)
    weights_t = torch.tensor(WW.flatten(), dtype=torch.float64, device=device)
    return coords_t, weights_t

def evaluate_gauss_integral(model: torch.nn.Module,
                            fourier_fn,
                            n: int,
                            device: torch.device,
                            integrand_fn) -> float:
    """
    Evaluate integral over [0,1]^2 using Gauss-Legendre tensor-product with n nodes per dim.
    - integrand_fn(model, feats, coords) -> scalar field at coords (shape (n*n,) or (n*n,1))
    - returns float integral value (Python float)
    """
    coords, weights = gauss_legendre_2d(n, device)
    feats = fourier_fn(coords)
    with torch.no_grad():
        vals = integrand_fn(model, feats, coords).view(-1)  # (n*n,)
    # weights are for integral over domain [0,1]^2
    integral = (vals * weights).sum().item()
    return integral

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
    lossi = None,
    lossm = None,
    tracker=L2Tracker()
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
        line_search_fn=line_search_fn,
        tolerance_grad=1e-9,
        tolerance_change=1e-12
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
        plt.plot(tracker.L2_history)
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel(r'$\mathcal{force}$', font={'family': 'Arial', 'size': 14})
        # plt.title('L2 Convergence during L-BFGS')
        # plt.grid(True)
        plt.tight_layout()
        plt.show()
    print(len(lossi))
    plt.figure()
    plt.plot(lossi)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font={'family': 'Arial', 'size': 14})
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(lossm)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font={'family': 'Arial', 'size': 14})
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    return loss_final.item(), loss_r_final.item(), loss_penalty_final.item(), mean_final.item(), tracker.L2_history

def spectral_derivative_2d(phi2d, dx, order_x=0, order_y=0):
    """
    phi2d: [Ny, Nx] real tensor
    dx: grid spacing (assume dx == dy)
    order_x, order_y: derivative orders
    returns real tensor [Ny, Nx]
    """
    Ny, Nx = phi2d.shape
    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=dx).to(phi2d.device)  # angular freq (rad/unit)
    ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=dx).to(phi2d.device)
    Kx, Ky = torch.meshgrid(kx, ky)  # [Ny, Nx]
    phi_hat = torch.fft.fft2(phi2d)
    factor = (1j * Kx) ** order_x * (1j * Ky) ** order_y
    deriv_hat = factor * phi_hat
    deriv = torch.fft.ifft2(deriv_hat).real
    return deriv

def lowpass_filter_2d(phi2d, dx, cutoff_cycles=None, sigma_cycles=None, mode='gaussian'):
    """
    Low-pass filter phi2d in frequency domain.
      - cutoff_cycles: cutoff in cycles per unit (e.g. cutoff_cycles=3 means keep |k| <= 3)
      - sigma_cycles: gaussian std in cycles per unit (if mode='gaussian' you can set sigma_cycles)
      - mode: 'gaussian' (smooth) or 'ideal' (hard mask)
    Returns filtered phi2d (real).
    """
    Ny, Nx = phi2d.shape
    # frequency in cycles per unit
    fx = torch.fft.fftfreq(Nx, d=dx).to(phi2d.device)  # cycles/unit
    fy = torch.fft.fftfreq(Ny, d=dx).to(phi2d.device)
    FX, FY = torch.meshgrid(fx, fy)  # [Ny, Nx]
    Kcycles = torch.sqrt(FX**2 + FY**2)  # cycles per unit

    Phi_hat = torch.fft.fft2(phi2d)

    if mode == 'ideal':
        assert cutoff_cycles is not None, "cutoff_cycles required for ideal filter"
        mask = (Kcycles <= float(cutoff_cycles)).to(phi2d.dtype)
        Phi_hat_filtered = Phi_hat * mask
    else:  # gaussian
        # if sigma_cycles not provided, derive from cutoff_cycles (broader if needed)
        if sigma_cycles is None:
            if cutoff_cycles is None:
                raise ValueError("Either cutoff_cycles or sigma_cycles must be provided")
            sigma_cycles = float(cutoff_cycles) * 0.5
        # gaussian in cycles space
        filt = torch.exp(-0.5 * (Kcycles / float(sigma_cycles))**2)
        Phi_hat_filtered = Phi_hat * filt

    phi_filtered = torch.fft.ifft2(Phi_hat_filtered).real
    return phi_filtered

# ---------- 示例：替换你脚本里最后计算 force 的那段 ----------
# 假定：
#   N = 200
#   device, dtype 与 model 一致
#   fourier, model 可用
# 我把流程写成函数，方便你直接调用

def compute_force_with_lowpass(model, fourier, N=200, cutoff_cycles=3.0,
                               sigma_cycles=None, filter_mode='gaussian',
                               kappa2=(0.01**2), device='cuda', dtype=torch.float64):
    """
    Returns: force (float), and optionally intermediate arrays if you want to inspect.
    - cutoff_cycles: integer-ish number of cycles per unit to keep (try 2~6)
    - sigma_cycles: for gaussian smoothing; if None, uses cutoff*0.5
    - filter_mode: 'gaussian' or 'ideal'
    """
    dx = 1.0 / N
    # build grid coords (consistent with your get_interior_points order)
    xs = torch.linspace(0.0, 1.0 - dx, N, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0 - dx, N, device=device, dtype=dtype)
    Xg, Yg = torch.meshgrid(xs, ys)  # shape [N, N]
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)  # [N*N,2]

    # model forward (no grad)
    coordsf = fourier(coords.to(device)).to(dtype)
    with torch.no_grad():
        phi_grid = model(coordsf).view(N, N)  # [N,N]

    # 1) low-pass filter phi_grid
    phi_filtered = lowpass_filter_2d(phi_grid, dx, cutoff_cycles=cutoff_cycles,
                                     sigma_cycles=sigma_cycles, mode=filter_mode)

    # 2) spectral derivatives on filtered phi
    phi_x = spectral_derivative_2d(phi_filtered, dx, order_x=1, order_y=0)
    phi_y = spectral_derivative_2d(phi_filtered, dx, order_x=0, order_y=1)
    phi_xxx = spectral_derivative_2d(phi_filtered, dx, order_x=3, order_y=0)
    phi_yyy = spectral_derivative_2d(phi_filtered, dx, order_x=0, order_y=3)

    # 3) compute g_x, g_y (per your formula)
    g_x = - kappa2 * phi_xxx + (3.0 * phi_filtered ** 2 - 1.0) * phi_x
    g_y = - kappa2 * phi_yyy + (3.0 * phi_filtered ** 2 - 1.0) * phi_y

    # 4) weighted norm (domain area =1) -> use mean for uniform grid
    integrand = g_x**2 + g_y**2
    I = integrand.mean()  # approximate integral over unit square
    force = float(torch.sqrt(I).item())
    return force, dict(phi_grid=phi_grid, phi_filtered=phi_filtered,
                       g_x=g_x, g_y=g_y, integrand=integrand)

def main():
    """
    # 1. 读取Excel文件
    file_path = 'data2dim.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(file_path, header=None)  # 假设没有标题行

    # 2. 转换为二维数组

    data = df.values

    # 3. 创建[0,1]范围的坐标网格
    x1 = torch.linspace(0, 1, data.shape[0])
    x2 = torch.linspace(0, 1, data.shape[1])
    X, Y = torch.meshgrid(x1, x2)  # 注意 PyTorch >=1.10 推荐加 indexing
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    X_sup = coords
    Y_sup = 0.02 + 0.01 * torch.cos(6* math.pi * X) * torch.cos(6 * math.pi * Y)
    # Y_sup = torch.where(
    #     (Y >= 0.25) & (Y <= 0.75) & (X >= 0) & (X <= 0.5),
    #     torch.tensor(1.0),
    #     torch.tensor(-1.0))
    # Y_sup = torch.tensor(data, dtype=torch.float32)
    print(Y_sup.mean().item())
    print(Y_sup.min().item(), Y_sup.max().item())

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(Y_sup, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.title('test')
    plt.show()
    """
    # torch.set_default_dtype(torch.float64)  # 全局改为 double
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) 创建 Fourier 映射
    ler = 1e-3
    freq = 3
    # fourier = FourierFeature(in_features=2, half_mapping_size=4, scale=2).to(device)
    # fourier = TorusFeature(in_dim=2, max_harmonic=4).to(device) cartesian
    fourier = FourierSeriesFeature(in_dim=2, max_freq=freq, mode="separable", normalize=True).to(device)
    # 2 * fourier.K.shape[0]
    # 1) 网络实例化：用 GlobalResNet 替换 drrnn
    net = GlobalResNet(in_dim= 2 * fourier.K.shape[0], width=100, out_dim=1, depth=6, phi=PowerReLU).to(device) #phi=nn.tanh
    model = ScaledBiasModel(net, target_mean=0.02, init_scale=0.05).to(device)
    # state_dict = torch.load("ritz_lgb.mdl", map_location=device)
    # model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=ler)
    print(f'max_freq={freq}, lr={ler}')
    # model.apply(init_weights)

    # result = fit_initial_output_2d_ls_then_finetune(model, X_sup, Y_sup, ff=fourier,
    #                                                 device=device, ridge=1e-8,
    #                                                 finetune=True, ft_lr=1e-4, ft_steps=1000)
    # # print(result)
    #   #kaiming
    # print(model)
    with torch.no_grad():
        coords = get_interior_points().to(device)  # [1002001, 2]
        coords_feat = fourier(coords) # [1002001, 2*mapping_size]
        # 模型预测
        pred = model(coords_feat).cpu().numpy().reshape(202, 202)
    print('first mean', pred.mean())
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





    best_loss, best_epoch = 2000, 0
    energy = 0
    losses = []
    losses_r1 = []
    losses_b1 = []
    egy = 0
    p_count = 0
    lr_flag = False
    # 超参数
    outer_iters = 10 # 外循环次数
    inner_iters = 100  # 每次内循环步数
    mu0 = 1 # 初始 penalty 系数
    rho = 1.2  # 每轮 mu 增长倍数
    mu_max = 2.0  # mu 上限
    target_mean = 0.02#pred.mean()
    # 初始化拉格朗日乘子和 mu
    lambda_param = 0.0
    mu = mu0

    xn = get_interior_points(N=200).to(device)
    xn_f = fourier(xn)

    tracker = L2Tracker(Nx=200, device=device, dtype=torch.float64)
    # 外循环：更新乘子 lambda
    for k in range(outer_iters):
        # 内循环：只用 penalty 项优化 u
        # x_r_raw = sample_sobol_interior(30000, device)
        for t in range(inner_iters):
            model.train()
            optimizer.zero_grad()

            # === 采样原始点 ===
            x_r_raw = get_interior_points(N=202).to(device)  # [N,2]
            x_r_raw.requires_grad_(True)
            # === Fourier 映射 ===
            x_r = fourier(x_r_raw)  # [N, 2*mapping_size]

            # 网络输出
            out_r = model(x_r)
            # print(out_r.shape)
            phi = out_r.view(-1)  # shape [N]
            # PDE 残差项
            # 1) 一阶导: grads [N, d]
            grads = torch.autograd.grad(outputs=phi,
                                        inputs=x_r_raw,
                                        grad_outputs=torch.ones_like(phi),
                                        create_graph=True)[0]  # [N, d]
            # grads = autograd.grad(outputs=phi.sum(), inputs=x_r_raw, create_graph=True)[0]
            # print(grads.shape)
            term1 = 0.5 * (0.01 ** 2) * torch.sum(grads ** 2, dim=1)
            term2 = 0.25 * (phi ** 2 - 1) ** 2
            loss_r = (term1 + term2).mean()


            # penalty 项
            mean_r = out_r.mean()
            constraint = mean_r - target_mean
            loss_penalty = 0.5 * mu * constraint ** 2


            # 总损失：
            loss = loss_r + loss_penalty + lambda_param * constraint
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                phi = model(xn_f)
            tracker.update(phi)

            if k > int(3 * outer_iters / 4):
                if torch.abs(loss) < best_loss:
                    best_loss = torch.abs(loss).item()
                    energy = loss_r.item()
                    best_epoch = t + k * inner_iters
                    torch.save(model.state_dict(), 'ritz_tmp.mdl')
            # if k == t == 0:
            #    print(out_r)
            #    print(loss_r.item())
            losses_r1.append(loss_r.item())
            losses_b1.append(loss_penalty.item())
            if t!=0 and t % 99 == 0:
                print(f"epoch:{t + 1 + k * inner_iters}, loss_r:{loss_r.item()}, loss_penalty:{loss_penalty.item()}")
        # 内循环结束，计算当前 mean
        model.eval()
        with torch.no_grad():
            out_mean = model(xn_f)
            mean_hat = out_mean.mean().item()
        # 外循环更新 lambda 和 mu
        # 在外循环中，当你已计算 mean_hat:
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
        #epoch:{(1 + k) * inner_iters}:
        model.eval()
        # L2sq = evaluate_gauss_integral(model, fourier, n=80, device=device, integrand_fn=lambda m,f,c:m(f).view(-1)**2)
        # L2 = float(np.sqrt(L2sq))
        # print('Gauss:', L2)

        with torch.no_grad():
            coords = get_interior_points().to(device)  # [1002001, 2]
            coords_feat = fourier(coords)  # [1002001, 2*mapping_size]
            # 模型预测
            pred = model(coords_feat).cpu().numpy().reshape(202, 202)
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1)
        h = plt.imshow(pred, interpolation='nearest', cmap='viridis',
                       extent=[0, 1, 0, 1],
                       origin='lower', aspect='auto',vmin=-1, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax)
        plt.title(f'epoch:{(k+1) * inner_iters} Output')
        plt.show()
    # call refine (把 lambda_param, mu 用你当前外循环的值填入)
    # 调用 L-BFGS refine
    loss_final, loss_r_final, loss_penalty_final, mean_final, L2_history = lbfgs_refine(
        model=model,
        fourier=fourier,
        X_ref=xn.double(),
        Nx=200,
        target_mean=0.02,  # 你之前的目标均值
        eps=0.01,
        lambda_param=0,  # 可根据你的增广拉格朗日调整
        mu=mu,  # 增广项系数
        device='cuda',
        use_double=True,
        lbfgs_lr=0.8,
        max_iter=200,
        history_size=10,
        line_search_fn='strong_wolfe',
        verbose=True,
        lossi = losses_r1,
        lossm = losses_b1,
        tracker=tracker
    )

    torch.save(model.state_dict(), 'ritz_lgb.mdl')
    print("Final L2 history length:", len(L2_history))
    # test = pd.DataFrame(columns=None, data=losses_r1)
    """
    model.eval()
    # === 采样原始点 ===
    x_h1 = get_interior_points(N=200).to(device)  # [N,2]
    x_h1.requires_grad_(True)
    # === Fourier 映射 ===
    x_h1f = fourier(x_h1)  # [N, 2*mapping_size]

    # 网络输出
    out_r_h1 = model(x_h1f)
    # print(out_r.shape)
    phi_h1 = out_r_h1.view(-1)  # shape [N]
    # PDE 残差项
    # 1) 一阶导: grads [N, d]
    grads_h1 = torch.autograd.grad(outputs=phi_h1,
                                inputs=x_h1,
                                grad_outputs=torch.ones_like(phi_h1),
                                create_graph=True)[0]  # [N, d]
    phi_x = grads_h1[:, 0]  # [N]
    phi_y = grads_h1[:, 1]  # [N]

    # 第二阶导数
    phi_xx = torch.autograd.grad(outputs=phi_x,
                                 inputs=x_h1,
                                 grad_outputs=torch.ones_like(phi_x),
                                 create_graph=True)[0][:, 0]  # [N]
    phi_yy = torch.autograd.grad(outputs=phi_y,
                                 inputs=x_h1,
                                 grad_outputs=torch.ones_like(phi_y),
                                 create_graph=True)[0][:, 1]  # [N]

    # 三阶导数
    phi_xxx = torch.autograd.grad(outputs=phi_xx,
                                  inputs=x_h1,
                                  grad_outputs=torch.ones_like(phi_xx),
                                  create_graph=True)[0][:, 0]  # [N]
    phi_yyy = torch.autograd.grad(outputs=phi_yy,
                                  inputs=x_h1,
                                  grad_outputs=torch.ones_like(phi_yy),
                                  create_graph=True)[0][:, 1]  # [N]

    # 计算 grad_delta 和 force
    grad_delta_x = - (0.01 ** 2) * phi_xxx + (3.0 * phi_h1 ** 2 - 1.0) * phi_x  # [N]
    grad_delta_y = - (0.01 ** 2) * phi_yyy + (3.0 * phi_h1 ** 2 - 1.0) * phi_y  # [N]
    grad_delta = (grad_delta_x**2 + grad_delta_y**2).view(200, 200)
    xh1 = torch.linspace(0, 1, 200, device=device, dtype=torch.get_default_dtype())
    # integrate over x first (dim=1), then y (dim=0)
    I_x_gra = torch.trapz(grad_delta, x=xh1, dim=1)  # integrate along x -> shape (Nx,)
    I_gra = torch.trapz(I_x_gra, x=xh1)  # scalar
    force = torch.sqrt(I_gra).item()
    print('force:', force)

    grad_delta_2 = - (0.01 ** 2) * (phi_xx + phi_yy) + (phi_h1 ** 3) - phi_h1
    grad_delta_2 = (grad_delta_2 ** 2).view(200, 200)
    I_x_gra = torch.trapz(grad_delta_2, x=xh1, dim=1)  # integrate along x -> shape (Nx,)
    I_gra = torch.trapz(I_x_gra, x=xh1)  # scalar
    force = torch.sqrt(I_gra).item()
    print('force2:', force)
    # 参数
    """






    # test.to_csv('loss_r.csv')
    print('best epoch:', best_epoch, 'best loss:', best_loss, 'energy:', energy)
    # plt.figure()
    # plt.plot(losses[1:], color='red', lw=2)
    # plt.show()
    plt.plot(losses_r1, color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font = {'family': 'Arial', 'size': 14})
    # plt.title('loss_r')
    plt.savefig('loss_r.png')
    plt.show()
    plt.figure()

    plt.semilogy(losses_b1, color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font = {'family': 'Arial', 'size': 14})
    plt.savefig('loss_b.png')
    plt.show()

    # print('force:', tracker.L2_history)
    plt.semilogy(tracker.L2_history, color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{force}$', font = {'family': 'Arial', 'size': 14})
    plt.show()


    losses_r1 = np.array(losses_r1)
    # 保存为.npy文件
    np.save('lossi.npy', losses_r1)

    losses_b1 = np.array(losses_b1)
    # 保存为.npy文件
    np.save('lossm.npy', losses_b1)

    forces = np.array(tracker.L2_history)
    # 保存为.npy文件
    np.save('force.npy', forces)

    # plot figure
    model.load_state_dict(torch.load('ritz_lgb.mdl'))
    model.eval()
    print('load from ckpt!')
    with torch.no_grad():
        coords = get_interior_points().to(device)  # [1002001, 2]
        # 👉 Fourier 映射
        coords_feat = fourier(coords)  # [1002001, 2*mapping_size]

        # 模型预测
        pred = model(coords_feat).cpu().numpy().reshape(202, 202)
    plt.figure(figsize=(10, 8))
    print('mean:', pred.mean().item())
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, 0, 1],
                   origin='lower', aspect='auto',vmin=-1, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    plt.savefig('pred.png')
    plt.show()
    #设置范围

if __name__ == '__main__':
    start_time = time.time()

    # 执行你的代码

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"代码执行时间：{execution_time//60} 分,{execution_time%60:.1f}秒")
    print(f"代码执行时间：{execution_time:.2f}秒")

