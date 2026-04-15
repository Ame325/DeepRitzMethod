import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import random
import time
import plotly.graph_objects as go
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import map_coordinates  # for trilinear interpolation
from torch.optim import LBFGS
torch.pi = math.pi
# Try to solve the equation:

'''
新的残差网络
'''

class PowerReLU(nn.Module):
    def __init__(self, inplace=False, power=1.3):
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
    def __init__(self, in_dim=3, width=100, out_dim=1, depth=6, phi=PowerReLU):#phi=nn.Tanh
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
        if x.dtype != self.K.dtype:
            x = x.to(self.K.dtype)
        proj = 2 * math.pi * (x @ self.K.t().to(x.device))
        sinc = torch.sin(proj)
        cosc = torch.cos(proj)
        feats = torch.cat([sinc, cosc], dim=-1)  # [N, 2*M]
        if self.normalize:
            feats = feats / math.sqrt(feats.shape[-1])
        return feats

class L2Tracker:
    def __init__(self, Nx=64, device='cuda', dtype=torch.float64):
        """
        Nx: grid size per axis (assume unit cube [0,1]^3 and uniform grid)
        device, dtype: used for computations
        """
        self.Nx = Nx
        self.device = torch.device(device)
        self.dtype = dtype
        self.prev_phi = None
        self.L2_history = []         # raw L2 history
        self.L2_filtered_history = []  # filtered L2 history (if used)

    @torch.no_grad()
    def lowpass_filter_3d(self, phi3, cutoff_cycles=None, sigma_cycles=None, mode='gaussian'):
        """
        phi3: tensor [Nx, Nx, Nx] real, on self.device/dtype
        cutoff_cycles: cutoff in cycles per unit (e.g. 3 means keep |k| <= 3)
        sigma_cycles: gaussian std in cycles per unit (if gaussian)
        mode: 'gaussian' or 'ideal' (hard mask)
        returns phi_filtered (real tensor [Nx,Nx,Nx])
        """
        Nx = self.Nx
        dx = 1.0 / Nx
        # frequency axes in cycles per unit
        fx = torch.fft.fftfreq(Nx, d=dx).to(self.device)
        fy = torch.fft.fftfreq(Nx, d=dx).to(self.device)
        fz = torch.fft.fftfreq(Nx, d=dx).to(self.device)
        KX, KY, KZ = torch.meshgrid(fx, fy, fz)  # shape [Nx,Nx,Nx]
        Kmag = torch.sqrt(KX**2 + KY**2 + KZ**2)

        Phi_hat = torch.fft.fftn(phi3)

        if mode == 'ideal':
            assert cutoff_cycles is not None, "cutoff_cycles required for ideal filter"
            mask = (Kmag <= float(cutoff_cycles)).to(phi3.dtype)
            Phi_hat_f = Phi_hat * mask
        else:  # gaussian
            if sigma_cycles is None:
                if cutoff_cycles is None:
                    raise ValueError("Either cutoff_cycles or sigma_cycles must be provided")
                sigma_cycles = float(cutoff_cycles) * 0.5
            filt = torch.exp(-0.5 * (Kmag / float(sigma_cycles))**2)
            Phi_hat_f = Phi_hat * filt

        phi_f = torch.fft.ifftn(Phi_hat_f).real
        return phi_f

    @torch.no_grad()
    def update(self, phi, do_filter=True, cutoff_cycles=3.0, sigma_cycles=None, filter_mode='gaussian'):
        """
        phi: model output, shape (N^3,) or (N^3,1) or (Nx,Nx,Nx)
        do_filter: whether to compute filtered-L2 (and store it)
        cutoff_cycles/sigma_cycles/filter_mode: filtering params
        Returns:
            L2_raw (float or None if first call),
            L2_filtered (float or None if not computed or first call)
        """
        # flatten and cast
        phi_flat = phi.view(-1).to(self.dtype).to(self.device)
        Np = self.Nx ** 3
        if phi_flat.numel() != Np:
            raise AssertionError(f"phi length {phi_flat.numel()} != {Np}")
        phi3 = phi_flat.view(self.Nx, self.Nx, self.Nx)

        # first call: init prev and return None(s)
        if self.prev_phi is None:
            self.prev_phi = phi3.clone()
            self.L2_history.append(None)
            if do_filter:
                self.L2_filtered_history.append(None)
            return None, None

        # raw L2 (uniform grid: integral ≈ mean * volume; volume=1)
        diff = phi3 - self.prev_phi
        mean_sq = torch.mean(diff * diff)
        L2_raw = float(torch.sqrt(mean_sq).item())
        self.prev_phi = phi3.clone()
        self.L2_history.append(L2_raw)

        L2_filtered = None
        if do_filter:
            # filter current phi and previous phi with same filter, then compute L2 of filtered diff
            phi_curr = phi3
            # Note: we could cache prev filtered phi, but for correctness filter both
            phi_prev = self.L2_filtered_prev if hasattr(self, 'L2_filtered_prev') else None

            phi_f_curr = self.lowpass_filter_3d(phi_curr, cutoff_cycles=cutoff_cycles,
                                                sigma_cycles=sigma_cycles, mode=filter_mode)
            # for prev, if not present, compute from prev_phi
            if phi_prev is None:
                phi_f_prev = self.lowpass_filter_3d(self.L2_history_prev_tensor() if hasattr(self, 'L2_history_prev_tensor') else self.prev_phi,
                                                    cutoff_cycles=cutoff_cycles, sigma_cycles=sigma_cycles, mode=filter_mode)
            else:
                phi_f_prev = phi_prev

            # compute filtered diff against last filtered prev stored in instance if available
            # Simpler: compute filtered(prev_phi) now (prev_phi stored before we overwrote it)
            # We need the previous phi (before we set self.prev_phi to current). To do that, we stored it earlier:
            # Instead of the convoluted above, let's compute filtered(prev) by applying filter to the difference of newly assigned prev stored in attribute 'last_prev_for_filter'
            # For clarity, we will recompute filtered(prev) from the last raw prev value stored in attribute self._last_prev_raw
            if hasattr(self, '_last_prev_raw'):
                phi_f_prev = self.lowpass_filter_3d(self._last_prev_raw, cutoff_cycles=cutoff_cycles,
                                                    sigma_cycles=sigma_cycles, mode=filter_mode)
            else:
                # fallback: compute from self.prev_phi_clone which currently holds phi3 (we already updated self.prev_phi)
                # To avoid complexity, compute filtered diff using filtered current and filtered of (previous stored in attribute),
                # but if not available, we approximate by filtering current and using filtered current shifted (this is unlikely).
                phi_f_prev = self.lowpass_filter_3d(self.prev_phi, cutoff_cycles=cutoff_cycles,
                                                    sigma_cycles=sigma_cycles, mode=filter_mode)

            # compute filtered difference: note we want diff between current filtered and previous filtered (previous was before update)
            # To ensure we have the correct previous raw phi, we should have stored it. Simplify by keeping a backup: store raw_prev before overwriting.
            # Implement by using an attribute _raw_prev that is set right before update writes to self.prev_phi.
            # But since we already overwrote prev, handle it more cleanly by changing logic: we'll compute filtered prev from history (last raw stored)
            # To implement safely, we refactor: store last raw phi in attribute self._last_raw_phi prior to overwriting; here we use that.
            # If not present, compute filtered diff as filtered(current) - filtered(current) -> zero (not ideal). So to be robust, ensure _last_raw_phi exists.

            # ----- simpler robust implementation below -----
            # We assume attribute self._last_raw_phi holds the previous raw phi (set before calling update).
            if hasattr(self, '_last_raw_phi'):
                phi_f_prev = self.lowpass_filter_3d(self._last_raw_phi, cutoff_cycles=cutoff_cycles,
                                                    sigma_cycles=sigma_cycles, mode=filter_mode)
            else:
                # worst-case compute filtered prev from self.prev_phi_clone which we saved at beginning (but in this code flow we didn't)
                phi_f_prev = self.lowpass_filter_3d(torch.zeros_like(phi3), cutoff_cycles=cutoff_cycles,
                                                    sigma_cycles=sigma_cycles, mode=filter_mode)

            diff_f = phi_f_curr - phi_f_prev
            mean_sq_f = torch.mean(diff_f * diff_f)
            L2_filtered = float(torch.sqrt(mean_sq_f).item())
            self.L2_filtered_history.append(L2_filtered)
            # store last filtered current for next call
            self.L2_filtered_prev = phi_f_curr.clone()

        # store last raw phi for next filter computation
        self._last_raw_phi = phi3.clone()

        return L2_raw, L2_filtered

    def reset(self):
        self.prev_phi = None
        self.L2_history = []
        self.L2_filtered_history = []
        if hasattr(self, '_last_raw_phi'):
            del self._last_raw_phi
        if hasattr(self, 'L2_filtered_prev'):
            del self.L2_filtered_prev

"""
class L2Tracker:
    def __init__(self, Nx=64, device='cuda', dtype=torch.float64):
        self.Nx = Nx
        self.device = device
        self.dtype = dtype
        self.prev_phi = None
        self.L2_history = []

    @torch.no_grad()
    def update(self, phi):
        # phi: (N^3,1) or (N^3,)
        phi_flat = phi.view(-1).to(self.dtype)
        Np = self.Nx**3
        assert phi_flat.numel() == Np, f"phi length {phi_flat.numel()} != {Np}"
        phi3 = phi_flat.view(self.Nx, self.Nx, self.Nx)

        if self.prev_phi is None:
            self.prev_phi = phi3.clone()
            return None

        diff = phi3 - self.prev_phi
        mean_sq = torch.mean(diff * diff)
        L2 = float(torch.sqrt(mean_sq).item())
        self.prev_phi = phi3.clone()
        self.L2_history.append(L2)
        return L2

    def reset(self):
        self.prev_phi = None
        self.L2_history = []
"""

def get_interior_points(resolution=25):
    """
    使用分层采样在三维空间 [0,1]×[0,1]×[0,1] 中创建点
    每个维度分成resolution个区间，然后在每个小立方体中随机采样一个点
    """
    # 计算每个维度的区间
    step = 1.0 / resolution

    # 生成每个小立方体的左下角坐标
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    z = torch.linspace(0, 1, resolution)
    # 创建网格
    X, Y, Z = torch.meshgrid(x, y, z)
    # 在每个小立方体中随机采样一个点
    coords = torch.stack([
        X.flatten() + torch.rand(resolution**3) * step,
        Y.flatten() + torch.rand(resolution**3) * step,
        Z.flatten() + torch.rand(resolution**3) * step
    ], dim=-1)
    return coords


def get_interior_points_sobol(N, device='cpu', dtype=torch.float32, seed=None):
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    return engine.draw(n=N).to(device=device, dtype=dtype)

def get_draw_points1(N=100):
    """
    在三维空间 [0,1]×[0,1]×[0,1] 中创建均匀网格点
    """
    # 定义三个维度上的坐标
    x1 = torch.linspace(0, 1, N)  # x轴
    x2 = torch.linspace(0, 1, N)  # y轴
    x3 = torch.linspace(0, 1, N)  # z轴

    # 创建三维网格
    X, Y, Z = torch.meshgrid(x1, x2, x3)

    # 将网格展平并堆叠成坐标点
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    return coords

def make_regular_grid(nx, ny, nz, x0=0.0, x1=1.0, y0=0.0, y1=1.0, z0=0.0, z1=1.0):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')  # shapes (nx,ny,nz)
    pts = np.stack([X.ravel(order='C'), Y.ravel(order='C'), Z.ravel(order='C')], axis=1)
    return pts.astype(np.float32), xs, ys, zs

def eval_model_on_grid(model, fourier, pts_np, nx, ny, nz,
                       device=torch.device('cuda'), batch_size=200000, dtype=torch.float32):
    """
    Evaluate model on pts_np (numpy (N,3)), in batches.
    Returns vol shaped (nz, ny, nx)
    """
    model = model.to(device)
    model.eval()
    N = pts_np.shape[0]
    preds = np.empty((N,), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Evaluating grid"):
            block = pts_np[i:i+batch_size]
            xb = torch.from_numpy(block).to(device=device, dtype=dtype)
            xb_mapped = fourier(xb)  # expects torch tensor on device
            out = model(xb_mapped)
            if isinstance(out, tuple):
                out = out[0]
            out = out.view(-1).cpu().numpy().astype(np.float32)
            preds[i:i+len(block)] = out
    vol_xyz = preds.reshape((nx, ny, nz), order='C')  # (nx,ny,nz)
    vol = np.transpose(vol_xyz, (2, 1, 0))  # -> (nz, ny, nx)
    return vol

def plotly_isosurface_and_slices(vol, xs, ys, zs, isovalue=0.0, slice_z_indices=None,
                                 cmap='Viridis', output_html=None, title=None):
    nz, ny, nx = vol.shape
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vol.ravel(order='C')
    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=xv, y=yv, z=zv, value=vals,
        isomin=isovalue, isomax=isovalue,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=cmap,
        showscale=True,
        name=f'isosurface={isovalue}'
    ))
    if slice_z_indices is None:
        slice_z_indices = [nz // 2]
    for k in slice_z_indices:
        if k < 0 or k >= nz:
            continue
        zval = zs[k]
        slice_arr = vol[k]  # (ny, nx)
        X2, Y2 = np.meshgrid(xs, ys, indexing='xy')
        Z2 = np.full_like(X2, zval)
        fig.add_trace(go.Surface(
            x=X2, y=Y2, z=Z2, surfacecolor=slice_arr,
            colorscale=cmap, cmin=vol.min(), cmax=vol.max(),
            showscale=False, opacity=0.9, name=f'slice z={zval:.4f}'
        ))
    fig.update_layout(scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='z',
        aspectmode='data'),
        width=1000, height=800,
        title=title if title is not None else "Isosurface & Slices"
    )
    fig.show()
    if output_html:
        import plotly.offline as pyo
        pyo.plot(fig, filename=output_html, auto_open=False)
        print(f"Saved interactive plot to {output_html}")
    return fig

def plot_surface_with_slices_and_box(vol, xs, ys, zs, isovalue=0.0,
                                     slice_z_indices=None, mesh_color='seagreen',
                                     mesh_opacity=0.92, slice_opacity=0.85,
                                     box_opacity=0.12, simplify_faces=None,
                                     output_html=None, title=None):
    """
    vol: (nz, ny, nx)
    xs, ys, zs: 1D arrays length nx, ny, nz
    slice_z_indices: list of z indices to show planar slices (0..nz-1)
    simplify_faces: if not None, target number of faces to simplify mesh (requires trimesh)
    """
    nz, ny, nx = vol.shape

    # marching_cubes: returns verts in (z, y, x) voxel index coordinates
    verts, faces, normals, values = measure.marching_cubes(vol, level=isovalue, spacing=(1.0,1.0,1.0))

    # optional simplification
    if simplify_faces is not None:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh = mesh.simplify_quadratic_decimation(int(simplify_faces))
            verts = mesh.vertices
            faces = mesh.faces
        except Exception as e:
            print("trimesh simplification failed:", e)

    # Map verts indices to physical coordinates (interpolate)
    ix = np.arange(nx); iy = np.arange(ny); iz = np.arange(nz)
    x_coords = np.interp(verts[:,2], ix, xs)
    y_coords = np.interp(verts[:,1], iy, ys)
    z_coords = np.interp(verts[:,0], iz, zs)

    i = faces[:,0].astype(np.int32)
    j = faces[:,1].astype(np.int32)
    k = faces[:,2].astype(np.int32)

    # cube vertices (outer box)
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]
    zmin, zmax = zs[0], zs[-1]
    cube_verts = np.array([
        [xmin,ymin,zmin],
        [xmax,ymin,zmin],
        [xmax,ymax,zmin],
        [xmin,ymax,zmin],
        [xmin,ymin,zmax],
        [xmax,ymin,zmax],
        [xmax,ymax,zmax],
        [xmin,ymax,zmax],
    ])
    cube_faces = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [1,2,6],[1,6,5],
        [2,3,7],[2,7,6],
        [3,0,4],[3,4,7],
    ])

    fig = go.Figure()

    # isosurface mesh (Mesh3d)
    fig.add_trace(go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i, j=j, k=k,
        color=mesh_color, opacity=mesh_opacity,
        flatshading=False, name=f'isosurface {isovalue}', hoverinfo='skip'
    ))

    # outer box
    fig.add_trace(go.Mesh3d(
        x=cube_verts[:,0], y=cube_verts[:,1], z=cube_verts[:,2],
        i=cube_faces[:,0], j=cube_faces[:,1], k=cube_faces[:,2],
        color='royalblue', opacity=box_opacity, flatshading=True, hoverinfo='skip', name='domain box'
    ))

    # add slices (Surface)
    if slice_z_indices is None:
        slice_z_indices = [nz//2]
    for kidx in slice_z_indices:
        if 0 <= kidx < nz:
            zval = zs[kidx]
            X2, Y2 = np.meshgrid(xs, ys, indexing='xy')  # shape (ny,nx)
            slice_arr = vol[kidx]  # (ny, nx)
            fig.add_trace(go.Surface(
                x=X2, y=Y2, z=np.full_like(X2, zval),
                surfacecolor=slice_arr,
                colorscale='Viridis', cmin=vol.min(), cmax=vol.max(),
                showscale=True, opacity=slice_opacity, name=f'slice z={zval:.3f}'
            ))

    fig.update_layout(
        scene=dict(xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z'), aspectmode='data'),
        width=1200, height=900, title=title or f"Isosurface (level={isovalue}) with slices"
    )

    fig.show()
    if output_html:
        import plotly.offline as pyo
        pyo.plot(fig, filename=output_html, auto_open=False)
        print(f"Saved interactive plot to {output_html}")
    return fig

def plot_marching_cubes_mesh_only(vol, xs, ys, zs, isovalue=0.0,
                                  mesh_color='seagreen', mesh_opacity=0.95,
                                  simplify_faces=None, output_html=None, title=None):
    """
    Render only the full 3D isosurface (no slices).
    vol shape: (nz, ny, nx)
    """
    nz, ny, nx = vol.shape
    # extract mesh (verts in voxel index coords z,y,x)
    verts, faces, normals, values = measure.marching_cubes(vol, level=isovalue, spacing=(1.0,1.0,1.0))
    # optional simplification (requires trimesh)
    if simplify_faces is not None:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh = mesh.simplify_quadratic_decimation(int(simplify_faces))
            verts = mesh.vertices
            faces = mesh.faces
        except Exception as e:
            print("trimesh simplify failed:", e)

    # map verts index coords -> physical coords via interpolation
    ix = np.arange(nx); iy = np.arange(ny); iz = np.arange(nz)
    x_coords = np.interp(verts[:,2], ix, xs)
    y_coords = np.interp(verts[:,1], iy, ys)
    z_coords = np.interp(verts[:,0], iz, zs)

    i = faces[:,0].astype(np.int32)
    j = faces[:,1].astype(np.int32)
    k = faces[:,2].astype(np.int32)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i, j=j, k=k,
        color=mesh_color, opacity=mesh_opacity,
        flatshading=False, name=f'isosurface {isovalue}', hoverinfo='skip'
    ))

    fig.update_layout(scene=dict(aspectmode='data',
                                 xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                      width=900, height=800,
                      title=title or f"Isosurface (level={isovalue})")
    fig.show()
    if output_html:
        import plotly.offline as pyo
        pyo.plot(fig, filename=output_html, auto_open=False)
    return fig

def plot_isosurface_plotly_only(vol, xs, ys, zs, isovalue=0.0, cmap='Viridis', output_html=None, title=None):
    """
    Simpler: use plotly.Isosurface directly to show full 3D (no slices).
    vol shape: (nz, ny, nx)
    """
    nz, ny, nx = vol.shape
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vol.ravel(order='C')
    fig = go.Figure(data=go.Isosurface(
        x=xv, y=yv, z=zv, value=vals,
        isomin=isovalue, isomax=isovalue, surface_count=1,
        caps=dict(x_show=False,y_show=False,z_show=False),
        colorscale=cmap, showscale=True, name=f'isosurface {isovalue}',
    ))
    fig.update_layout(scene=dict(aspectmode='data'), width=900, height=800, title=title or f"Plotly Isosurface {isovalue}")
    fig.show()
    if output_html:
        import plotly.offline as pyo
        pyo.plot(fig, filename=output_html, auto_open=False)
    return fig

def plot_volume_render(vol, xs, ys, zs, cmap='Viridis', opacity=0.1, surface_count=20, output_html=None, title=None):
    """
    vol: (nz, ny, nx)
    surface_count: controls number of isosurfaces used internally (affects visual quality)
    opacity: base opacity (0..1), volume uses low opacity typically
    """
    nz, ny, nx = vol.shape
    # flatten coordinates (must match ordering used to build vol)
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vol.ravel(order='C')
    vmin = float(vol.min())
    vmax = float(vol.max())
    fig = go.Figure(data=go.Volume(
        x=xv, y=yv, z=zv, value=vals,
        isomin=vmin, isomax=vmax,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=cmap,
        showscale=True
    ))
    fig.update_layout(scene=dict(aspectmode='data'), width=900, height=800, title=title or "Volume rendering")
    if output_html:
        fig.write_html(output_html, include_plotlyjs='cdn')  # 或 include_plotlyjs='directory'
        print(f"Plot saved to: {output_html}")
    else:
        fig.show()

def plot_multilevel_isosurfaces(vol, xs, ys, zs, n_levels=8, cmap='Viridis', output_html=None, title=None):
    """
    vol: (nz, ny, nx)
    n_levels: number of levels (surface_count)
    This uses Plotly.Isosurface with surface_count=n_levels to color by value.
    """
    nz, ny, nx = vol.shape
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vol.ravel(order='C')

    vmin = float(vol.min())
    vmax = float(vol.max())
    fig = go.Figure(data=go.Isosurface(
        x=xv, y=yv, z=zv, value=vals,
        isomin=vmin, isomax=vmax,
        surface_count=n_levels,
        colorscale=cmap,
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True
    ))
    fig.update_layout(scene=dict(aspectmode='data'), width=900, height=800, title=title or "Multilevel Isosurfaces")
    # 在生成 fig 后添加 volume trace（低 opacity）
    fig.add_trace(go.Volume(
        x=xv, y=yv, z=zv, value=vals,
        opacity=0.04, surface_count=20, colorscale='Viridis',
        cmin=0.0, cmax=1.0, showscale=False
    ))
    if output_html:
        fig.write_html(output_html, include_plotlyjs='cdn')  # 或 include_plotlyjs='directory'
        print(f"Plot saved to: {output_html}")
    else:
        fig.show()

def plot_volume_no_holes(vol, xs, ys, zs,
                         smooth_sigma=0.6,
                         normalize=True,
                         percentile_clip=(0.1, 99.9),
                         surface_count=28,
                         opacity_preset='middle',
                         cmap='Viridis',
                         save_html="volume_no_holes.html"):
    # vol: (nz, ny, nx)
    vol_proc = vol.copy()
    # optional smoothing to remove voxelization
    if smooth_sigma and smooth_sigma > 0:
        vol_proc = gaussian_filter(vol_proc, sigma=smooth_sigma)

    # percentile clip (very permissive because range is -1..1)
    low_pct, high_pct = percentile_clip
    vmin = float(np.percentile(vol_proc, low_pct))
    vmax = float(np.percentile(vol_proc, high_pct))
    if vmax <= vmin:
        vmin, vmax = float(vol_proc.min()), float(vol_proc.max())

    if normalize:
        vals_norm = (vol_proc - vmin) / (vmax - vmin)
        vals_norm = np.clip(vals_norm, 0.0, 1.0)
    else:
        vals_norm = vol_proc

    nz, ny, nx = vol_proc.shape
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vals_norm.ravel(order='C')

    # opacityscale: emphasize middle values (interface) but keep inside non-empty
    opacityscale = [
        [0.00, 0.00],
        [0.25, 0.00],
        [0.40, 0.15],
        [0.50, 0.75],
        [0.60, 0.15],
        [0.80, 0.00],
        [1.00, 0.00],
    ]

    trace_kwargs = dict(
        x=xv, y=yv, z=zv, value=vals,
        surface_count=surface_count,
        colorscale=cmap,
        cmin=0.0, cmax=1.0,
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True
    )
    trace_kwargs['opacity'] = 1.0
    trace_kwargs['opacityscale'] = opacityscale

    fig = go.Figure(data=go.Volume(**trace_kwargs))
    fig.update_layout(scene=dict(aspectmode='data'),
                      width=1000, height=900,
                      title="Volume rendering (no holes)")
    fig.write_html(save_html, include_plotlyjs='cdn')
    print("Saved volume html to", save_html)
    return fig


def plot_multilevel_isosurfaces_robust(
    vol, xs, ys, zs,
    preset='B',               # 'A'/'B'/'C' 三种预设
    smooth=True,              # 是否先用高斯滤波
    save_html=None,           # 文件名，如 "iso.html"
    cmap='Viridis',
    width=900, height=800
):
    """
    vol: numpy array (nz, ny, nx)
    xs, ys, zs: 1D arrays
    preset: 'A' conservative, 'B' stronger fill, 'C' aggressive
    Returns fig (plotly Figure)
    """
    # presets
    presets = {
        'A': {'surface_count': 10, 'opacity': 0.5, 'p_low': 1.0, 'p_high': 99.0, 'sigma': 0.4},
        'B': {'surface_count': 30, 'opacity': 0.3, 'p_low': 2.0, 'p_high': 98.0, 'sigma': 0.6},
        'C': {'surface_count': 30, 'opacity': 0.92, 'p_low': 5.0, 'p_high': 95.0, 'sigma': 0.8},
    }
    cfg = presets.get(preset, presets['B'])

    # 1) optional smoothing (helps connectivity)
    vol_proc = vol.copy()
    if smooth:
        vol_proc = gaussian_filter(vol_proc, sigma=cfg['sigma'])

    # 2) percentile clip to avoid outliers dominating
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin:
        # fallback to absolute min/max
        vmin, vmax = float(vol_proc.min()), float(vol_proc.max())
    # If dynamic range tiny, expand a bit
    if vmax - vmin < 1e-8:
        mid = 0.5*(vmin+vmax)
        vmin = mid - 1e-3
        vmax = mid + 1e-3

    # 3) normalize to 0..1 for consistent color/opacity handling
    vol_norm = (vol_proc - vmin) / (vmax - vmin)
    vol_norm = np.clip(vol_norm, 0.0, 1.0)

    # 4) prepare flattened coords (match ravel(order='C'))
    nz, ny, nx = vol_norm.shape
    xv = np.repeat(xs, ny * nz)
    yv = np.tile(np.repeat(ys, nz), nx)
    zv = np.tile(zs, nx * ny)
    vals = vol_norm.ravel(order='C')

    # 5) build isosurface
    fig = go.Figure(data=go.Isosurface(
        x=xv, y=yv, z=zv, value=vals,
        isomin=0.0, isomax=1.0,
        surface_count=cfg['surface_count'],
        colorscale=cmap,
        opacity=cfg['opacity'],
        caps=dict(x_show=False, y_show=False, z_show=False),
        cmin=0.0, cmax=1.0,
        showscale=True
    ))
    fig.update_layout(scene=dict(aspectmode='data'),
                      width=width, height=height,
                      title=f"Isosurfaces preset={preset}, sc={cfg['surface_count']}, op={cfg['opacity']}")
    # 6) save or show
    if save_html:
        fig.write_html(save_html, include_plotlyjs='cdn')
        print("Saved isosurface HTML to", save_html)
    else:
        fig.show()
    return fig


def mesh_colored_by_value(vol, xs, ys, zs, isovalue=None, smooth_sigma=None, cmap='Viridis', output_html=None):
    """
    Extract one isosurface (level = isovalue) and color the surface by the underlying vol values
    (interpolated at vertex positions).
    """
    nz, ny, nx = vol.shape
    if smooth_sigma is not None:
        from scipy.ndimage import gaussian_filter
        vol_proc = gaussian_filter(vol, sigma=smooth_sigma)
    else:
        vol_proc = vol

    vmin, vmax = float(vol_proc.min()), float(vol_proc.max())
    if isovalue is None:
        isovalue = 0.5 * (vmin + vmax)  # midpoint by default

    verts, faces, normals, values = measure.marching_cubes(vol_proc, level=isovalue, spacing=(1.0,1.0,1.0))
    # verts are in voxel-index coords (z,y,x). Map to physical coords:
    ix = np.arange(nx); iy = np.arange(ny); iz = np.arange(nz)
    x_coords = np.interp(verts[:,2], ix, xs)
    y_coords = np.interp(verts[:,1], iy, ys)
    z_coords = np.interp(verts[:,0], iz, zs)

    # Interpolate volume values at vert positions to get intensity (use index-space mapping)
    # Convert verts to index-space floats (z_idx, y_idx, x_idx)
    verts_idx = verts  # already in index coordinates due to spacing=(1,1,1)
    # map_coordinates expects array in (z,y,x) order
    coords = np.stack([verts_idx[:,0], verts_idx[:,1], verts_idx[:,2]], axis=0)
    # order=1 for trilinear interpolation
    vert_vals = map_coordinates(vol_proc, coords, order=1, mode='nearest')

    i = faces[:,0].astype(np.int32)
    j = faces[:,1].astype(np.int32)
    k = faces[:,2].astype(np.int32)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i, j=j, k=k,
        intensity=vert_vals,        # per-vertex scalar for coloring
        colorscale=cmap,
        showscale=True,
        colorbar=dict(title="value"),
        flatshading=False,
        opacity=1.0
    ))
    fig.update_layout(scene=dict(aspectmode='data'), width=900, height=800,
                      title=f"Surface colored by value (isovalue={isovalue:.6f})")
    if output_html:
        fig.write_html(output_html, include_plotlyjs='cdn')  # 或 include_plotlyjs='directory'
        print(f"Plot saved to: {output_html}")
    else:
        fig.show()

def get_draw_points(resolution=202, z_val=0):
    """
    简化的二维分层采样，然后添加z=0坐标
    """
    # 计算每个维度的区间
    step = 1.0 / resolution

    # 生成每个小方格的左下角坐标
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)

    # 创建网格
    X, Y = torch.meshgrid(x, y)

    # 在每个小方格中随机采样一个点
    coords_2d = torch.stack([
        X.flatten() + torch.rand(resolution ** 2) * step,
        Y.flatten() + torch.rand(resolution ** 2) * step
    ], dim=-1)

    # 添加z=0坐标，扩展为三维
    coords = torch.stack([
        X.flatten() + torch.rand(resolution**2) * step,
        Y.flatten() + torch.rand(resolution**2) * step,
        torch.ones(resolution**2) * z_val  # 使用指定的z值
    ], dim=-1)
    return coords

def choose_isovalue(vol, method='midpoint', percentile=50):
    """
    method: 'midpoint' | 'percentile' | 'manual'
    percentile: used if method == 'percentile'
    returns selected isovalue (float)
    """
    vmin = float(vol.min())
    vmax = float(vol.max())
    print(f"Volume range: min={vmin:.6g}, max={vmax:.6g}, mean={vol.mean():.6g}")
    if vmax - vmin < 1e-8:
        # nearly constant volume
        return float((vmin + vmax) / 2.0)
    if method == 'midpoint':
        return 0.5 * (vmin + vmax)
    elif method == 'percentile':
        return float(np.percentile(vol, percentile))
    else:
        raise ValueError("unknown method")


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
    lossm =None,
    tracker = L2Tracker()
):
    if lossi is None:
        lossi = []
    if lossm is None:
        lossm = []
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
    def lowpass_consistency_penalty(phi3, cutoff_cycles=3.0, device=None):
        """
        Memory-friendly, differentiable penalty: ||phi - lowpass(phi)||^2_mean
        phi3: [N,N,N] torch tensor
        """
        if device is None:
            device = phi3.device
        N = phi3.shape[0]
        dx = 1.0 / N
        # rfftn
        Phi_hat = torch.fft.rfftn(phi3)
        fx = torch.fft.fftfreq(N, d=dx).to(device)
        kz = torch.fft.rfftfreq(N, d=dx).to(device)
        KX = fx[:, None, None]
        KY = fx[None, :, None]
        KZ = kz[None, None, :]
        sigma = cutoff_cycles * 0.5
        filt = torch.exp(-0.5 * ((KX ** 2 + KY ** 2 + KZ ** 2) ** 0.5 / float(sigma)) ** 2).to(Phi_hat.dtype)
        Phi_hat_f = Phi_hat * filt
        # inverse rfftn to get filtered phi
        phi_low = torch.fft.irfftn(Phi_hat_f, s=(N, N, N))
        pen = torch.mean((phi3 - phi_low) ** 2)
        # free memory
        del Phi_hat, Phi_hat_f, phi_low, filt, KX, KY, KZ
        torch.cuda.empty_cache()
        return pen

    def lowpass_3d(phi3, cutoff_cycles=3.0, sigma=None):
        Nx = phi3.shape[0]
        dx = 1.0 / Nx
        Phi_hat = torch.fft.fftn(phi3)
        fx = torch.fft.fftfreq(Nx, d=dx).to(phi3.device)
        KX, KY, KZ = torch.meshgrid(fx, fx, fx)
        Kmag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        if sigma is None:
            sigma = cutoff_cycles * 0.5
        filt = torch.exp(-0.5 * (Kmag / float(sigma)) ** 2).to(Phi_hat.dtype)
        phi_f = torch.fft.ifftn(Phi_hat * filt).real
        return phi_f

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

    func_evals = {'count': 0}
    def closure():
        """
        lbfgs.zero_grad()

        # 保证 X 是 leaf tensor 且 requires_grad=True
        nonlocal X
        X = X.detach().clone().requires_grad_(True)

        loss, _, _, _, _, phi = compute_loss_on_X(X)
        loss.backward()

        L2_val,L2fi = tracker.update(phi, do_filter=True)
        if verbose and L2_val is not None:
            print(f"L2 between last two predictions: {L2_val:.5e},{L2fi:.5e}")
        return loss
    """
        func_evals['count'] += 1
        lbfgs.zero_grad()
        loss, loss_r_val, loss_penalty, mean_r, per_sample, phi = compute_loss_on_X(X)  # 你已有的函数
        loss.backward()
        L2_val, L2f = tracker.update(phi)
        if verbose and L2_val is not None:
            print(f"L2 between last two predictions: {L2_val:.5e}, {L2f:.5e}")
        """
        # 诊断输出（每次 closure 或每 5 次）
        if func_evals['count'] % 5 == 0:
            # grad norm
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.detach().norm().item()**2
            grad_norm = total_norm_sq**0.5

            # compute phi on full grid for L2 and spectrum (no grad)
            with torch.no_grad():
                phi_full = phi.view(-1)
                L2_raw, L2_filt = tracker.update(phi_full, do_filter=True, cutoff_cycles=3.0)
                # compute simple spectrum summary
                phi3 = phi_full.view(Nx,Nx,Nx).cpu().numpy()
                import numpy as np
                spec = np.abs(np.fft.fftn(phi3))
                spec_max, spec_mean = float(spec.max()), float(spec.mean())
            print(f"eval {func_evals['count']}: loss={loss.item():.3e}, loss_r={loss_r_val.item():.3e}, "
                  f"mean_r={mean_r.item():.3e}, grad_norm={grad_norm:.3e}, "
                  f"L2_raw={'N/A' if L2_raw is None else f'{L2_raw:.3e}'}, "
                  f"L2_filt={'N/A' if L2_filt is None else f'{L2_filt:.3e}'}, "
                  f"spec_max={spec_max:.3e}, spec_mean={spec_mean:.3e}")
        """
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

def lowpass_3d(phi3, dx, cutoff_cycles):
    # phi3: torch tensor [N,N,N]
    Phi_hat = torch.fft.fftn(phi3)
    # build frequency grids
    kx = torch.fft.fftfreq(phi3.shape[2], d=dx).to(phi3.device)
    ky = torch.fft.fftfreq(phi3.shape[1], d=dx).to(phi3.device)
    kz = torch.fft.fftfreq(phi3.shape[0], d=dx).to(phi3.device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz)
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)
    mask = (K <= cutoff_cycles).to(phi3.dtype)
    Phi_hat_filtered = Phi_hat * mask
    return torch.fft.ifftn(Phi_hat_filtered).real


def main():
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) 创建 Fourier 映射
    ler = 1e-3
    freq = 3
    # fourier = FourierFeature(in_features=2, half_mapping_size=4, scale=2).to(device)
    # fourier = TorusFeature(in_dim=2, max_harmonic=4).to(device)  cartesian separable
    mode = "separable"
    fourier = FourierSeriesFeature(in_dim=3, max_freq=freq, mode="separable", normalize=True).to(device)
    # 2 * fourier.K.shape[0]
    # 1) 网络实例化：用 GlobalResNet 替换 drrnn
    print('mode:', mode)
    # model = GlobalResNet(in_dim= 2 * fourier.K.shape[0], width=100, out_dim=1, depth=6, phi=PowerReLU).to(device) #phi=nn.tanh
    net = GlobalResNet(in_dim= 2 * fourier.K.shape[0], width=80, out_dim=1, depth=4, phi=PowerReLU).to(device) #phi=nn.tanh
    model = ScaledBiasModel(net, target_mean=0.02, init_scale=0.05).to(device)
    # state_dict = torch.load("ritz_3dlgb.mdl", map_location=device)
    # model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=ler)
    print(f'max_freq={freq}, lr={ler}')
    # model.apply(init_weights)
    # print(model)
    model.train()
    nx, ny, nz = 64, 64, 64
    batch_size = 10_000_000
    # batch_size = 50000
    output_html = "initial.html"
    # output_html1 = "initial1.html"
    vol_npy_path = "vol_3d.npy"

    # build gridssddd
    print("Building grid...")
    pts, xs, ys, zs = make_regular_grid(nx, ny, nz)
    print(f"Grid built: {nx}x{ny}x{nz} = {pts.shape[0]} points")
    # evaluate (this may take long)
    print("Evaluating model on grid (batched inference)...")
    vol = eval_model_on_grid(model, fourier, pts, nx, ny, nz, device=device, batch_size=batch_size)
    print('vol_min:',vol.min(),'vol_max:',vol.max())
    sigma = 0.6  # try 0.4~1.0; larger => smoother surface
    vol = gaussian_filter(vol, sigma=sigma)
    np.save('initial.npy', vol)
    plot_volume_render(vol, xs, ys, zs, cmap='Viridis', opacity=0.6, surface_count=5, output_html=output_html)
    # plot_multilevel_isosurfaces(vol, xs, ys, zs, n_levels=12, cmap='Viridis',output_html=output_html)
    # mesh_colored_by_value(vol, xs, ys, zs, isovalue=None, smooth_sigma=0.6, output_html=output_html)

    best_loss, best_epoch = 2000, 0
    energy = 0
    mean = 100
    losses = []
    losses_r1 = []
    losses_m1 = []
    un = []
    forces = []
    # 超参数
    outer_iters = 10  # 外循环次数
    inner_iters = 100  # 每次内循环步数
    mu0 = 1.0  # 初始 penalty 系数
    rho = 1.2  # 每轮 mu 增长倍数
    mu_max = 2.0  # mu 上限
    target_mean = 0.02#pred.mean()
    # 初始化拉格朗日乘子和 mu
    lambda_param = 0.0
    mu = mu0
    print(f'mu0:{mu0}, rho:{rho}, mu_max:{mu_max}')

    eps = 0.01  # or your eps variable
    N = 100000  # overall sample points (tune)
    # N = 25 * 25 * 25
    #batch_size = 20000  # tune: 4k/8k/16k... for 24GB GPU start with 8k
    # Sobol engine 在循环外创建（避免每次初始化开销）
    sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)

    xn = get_draw_points1(N=64).to(device)
    # xn = pts.to(device=device, dtype=torch.float32)
    xn_f = fourier(xn)

    tracker = L2Tracker(Nx=64, device=device, dtype=torch.float64)
    # 外循环：更新乘子 lambda
    for k in range(outer_iters):
        # 内循环：只用 penalty 项优化 u
        for t in range(inner_iters):
            model.train()
            # ------------- 1) 在 CPU 上用 Sobol 生成 N 点（不占显存） -------------
            pts = sobol_engine.draw(n=N).to(dtype=torch.float32, device='cpu')  # (N,3) in [0,1]^3
            # pts = get_interior_points()
            # pts = get_draw_points1(N=64).to(device)
            # ------------- 2) 用 no_grad 快速估计 mean (constraint_val) -------------
            # 保留用于 logging 的 detached 全样本均值估计（不保存图）
            sum_phi_detached = 0.0
            with torch.no_grad():
                model.eval()
                for i in range(0, N, batch_size):
                    xb = pts[i:i + batch_size].to(device=device, dtype=torch.float32)
                    xb_mapped = fourier(xb)  # (B, mapping_dim)
                    phi_b = model(xb_mapped).view(-1)  # (B,)
                    sum_phi_detached += float(phi_b.sum().cpu())
                model.train()
            mean_phi_val = sum_phi_detached / N
            constraint_val = mean_phi_val - target_mean  # python float (detached)

            # 为日志准备的常量（标量）
            loss_penalty_scalar = 0.5 * mu * (constraint_val ** 2)
            # total lambda term for logging would be lambda_param * constraint_val

            # ------------- 3) 分批计算并**累积可导量**（不立即 backward）-------------
            sum_energy_detached = 0.0  # 累计能量用于日志（python float）
            sum_phi_for_lambda_detached = 0.0  # 累积 phi_sum 用于日志（python float）
            first_batch_out_detached = None

            # 用于保留可导的全样本量（torch scalar）
            total_phi_sum = None  # will be torch scalar (requires grad through model outputs)
            energy_loss_for_backward = None  # will be torch scalar (normalized by N)

            for i in range(0, N, batch_size):
                xb = pts[i:i + batch_size].to(device=device, dtype=torch.float32)
                xb.requires_grad_(True)  # 仅对当前 batch 开启 grad
                xb_mapped = fourier(xb)  # (B, mapping_dim)

                out = model(xb_mapped).view(-1)  # (B,)

                # grads wrt physical coords xb (create_graph True to allow higher-order grads if needed)
                grads = autograd.grad(outputs=out,
                                      inputs=xb,
                                      grad_outputs=torch.ones_like(out),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_sq = (grads ** 2).sum(dim=1)  # (B,)

                term1 = 0.5 * (eps ** 2) * grad_sq  # (B,)
                term2 = 0.25 * (out ** 2 - 1.0) ** 2  # (B,)
                energy_density = term1 + term2  # (B,)

                # batch 的 loss_r 部分：按总 N 归一化（保证累加等价）
                batch_loss_r = energy_density.sum() / N  # torch scalar (keeps gradient path)

                # 累加为一个 torch scalar，最终统一 backward
                if energy_loss_for_backward is None:
                    energy_loss_for_backward = batch_loss_r
                else:
                    energy_loss_for_backward = energy_loss_for_backward + batch_loss_r

                # 累积 phi_sum 为 torch scalar（用于后面计算全样本 mean 的可导版本）
                phi_sum_batch = out.sum()  # torch scalar
                if total_phi_sum is None:
                    total_phi_sum = phi_sum_batch
                else:
                    total_phi_sum = total_phi_sum + phi_sum_batch

                # logging（detach 后）
                sum_energy_detached += float(energy_density.detach().sum().cpu())
                sum_phi_for_lambda_detached += float(phi_sum_batch.detach().cpu())

                if first_batch_out_detached is None:
                    first_batch_out_detached = out.detach().cpu().clone()

                # 清理局部变量（保留 total_phi_sum & energy_loss_for_backward）
                del xb, xb_mapped, out, grads, grad_sq, term1, term2, energy_density
                del phi_sum_batch
                torch.cuda.empty_cache()

            # ------------- 4) 在所有 batch 累积后一次性构造 penalty + lambda，并 backward 更新参数 -------------
            # total_phi_sum is a torch scalar with grad path; compute total (torch) constraint
            c_total = (total_phi_sum / N) - target_mean  # torch scalar, requires grad
            loss_penalty = 0.5 * mu * (c_total ** 2)
            loss_lambda = lambda_param * c_total  # lambda_param can be python float or 0-d tensor

            total_loss = energy_loss_for_backward + loss_penalty + loss_lambda

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ------------- 5) 计算并记录用于日志的标量（等价于原来的 loss_r, loss_penalty） -------------
            loss_r_scalar = sum_energy_detached / N
            loss_penalty_scalar = 0.5 * mu * ((mean_phi_val - target_mean) ** 2)  # detached scalar for logging
            loss_total_scalar = loss_r_scalar + loss_penalty_scalar + lambda_param * constraint_val

            model.eval()
            with torch.no_grad():
                phi = model(xn_f)
            L2_val = tracker.update(phi, do_filter=True)

            # 记录 / 保存 / 打印（保留你原有逻辑）
            # 更新 best loss 与保存模型（你原来是在后 1/4 区间开始保存）
            if k > int(3 * outer_iters / 4):
                if abs(loss_total_scalar) < best_loss:
                    best_loss = abs(loss_total_scalar)
                    energy = loss_r_scalar
                    mean = mean_phi_val
                    best_epoch = t + k * inner_iters
                    torch.save(model.state_dict(), 'ritz_tmp.mdl')

            # 原来在 k==t==0 打印 out_r 与 loss_r
            if k == 0 and t == 0:
                print(first_batch_out_detached.mean().item())  # numpy/tensor 打印
                print(loss_r_scalar)

            # append 到历史数组
            losses_r1.append(loss_r_scalar)
            losses_m1.append(loss_penalty_scalar)

            if t % 50 == 0:
                print(
                    f"epoch:{t + k * inner_iters}, loss_r:{loss_r_scalar}, loss_penalty:{loss_penalty_scalar:.6e}")

        # ========== 内循环结束，外循环更新 lambda 和 mu（使用评估 mean_hat） ==========
        # 这里用较小代价的 no_grad 重新估计 mean_hat（或复用上面 mean_phi_val）
        with torch.no_grad():
            out_mean = model(xn_f)
            mean_hat = out_mean.mean().item()
            # mean_hat = mean_phi
        lambda_param += mu * (mean_hat - target_mean)
        mu = min(mu * rho, mu_max)
        """
        phit = out_mean.view(64, 64, 64).cpu().numpy()
        # compute 3D FFT magnitude (use np.fft.fftn)
        spec = np.abs(np.fft.fftn(phit))
        spec_shift = np.fft.fftshift(spec)
        # inspect max and radial energy (you can plot slices)
        print("spec max:", spec_shift.max(), "mean:", spec_shift.mean())
        # optionally compute radial energy or low/high freq ratio
        # 外循环更新 lambda 和 mu
        # 在外循环中，当你已计算 mean_hat:
        constraint = mean_hat - target_mean
        abs_constraint = abs(constraint)

        # decide whether constraint improved compared to previous recorded
        improved = (prev_constraint / (abs_constraint + 1e-12)) > improve_ratio
        # 只在满足一定条件时更新 lambda，并用 relaxation + clipping
        # 例如：若 inner loop 足够做了，且 constraint 不是非常小
        if abs_constraint > tol_constraint:
            # relaxation 更新，避免一次性过大跳动
            lambda_param = lambda_param + tau * mu * constraint
            # 裁剪 lambda 防止发散
            lambda_param = max(min(lambda_param, lambda_clip), -lambda_clip)
            lr_flag = True

        # 只在 constraint 没有明显改善时温和增大 mu
        if not improved and (abs_constraint > tol_constraint):
            mu = min(mu * rho_increase, mu_max)
        # 如果有明显改善，可以不增大 mu，甚至按需要轻微减小或固定
        # else:
        #     mu = max(mu / 1.02, mu0)  # 可选：小幅减小或保持

        # 记录当前 constraint 以便下一次比较
        prev_constraint = max(abs_constraint, 1e-12)
        """
        print(f"Outer {k}: mean={mean_hat:.4f}, lambda={lambda_param:.4f}, mu={mu:.2f}")

        model.eval()
        with torch.no_grad():
            coords = get_draw_points().to(device)  # [1002001, 2]
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

    # 调用 L-BFGS refine
    loss_final, loss_r_final, loss_penalty_final, mean_final, L2_history = lbfgs_refine(
        model=model,
        fourier=fourier,
        X_ref=xn.double(),
        Nx=64,
        target_mean=0.02,  # 你之前的目标均值
        eps=0.01,
        lambda_param=0,  # 可根据你的增广拉格朗日调整
        mu=mu,  # 增广项系数
        device='cuda',
        use_double=True,
        lbfgs_lr=0.8,
        max_iter=50,
        history_size=40,
        line_search_fn='strong_wolfe',
        verbose=True,
        lossi = losses_r1,
        lossm = losses_m1,
        tracker=tracker
    )
    torch.save(model.state_dict(), 'ritz_3dlgb.mdl')
    # print("Final L2 history length:", len(L2_history))

    # test = pd.DataFrame(columns=None, data=losses_r1)
    # test.to_csv('loss_r.csv')
    print('best epoch:', best_epoch, 'best loss:', best_loss, 'energy:', energy, 'mean:', mean)
    # plt.figure()
    # plt.plot(losses[1:], color='red', lw=2)
    # plt.show()
    plt.figure()
    plt.plot(losses_r1, color='red', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_i$', font = {'family': 'Arial', 'size': 14})
    # plt.title('loss_r')
    plt.savefig('loss_r.png')
    plt.show()

    plt.figure()
    plt.semilogy(losses_m1, color='blue', lw=2)
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{L}_m$', font = {'family': 'Arial', 'size': 14})
    plt.savefig('loss_b.png')
    plt.show()

    # print('min force:', min(forces))
    plt.figure()
    plt.plot(tracker.L2_history)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{force}$', font={'family': 'Arial', 'size': 14})
    # plt.title('L2 Convergence during L-BFGS')
    # plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.plot(tracker.L2_filtered_history)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel(r'$\mathcal{force}$', font={'family': 'Arial', 'size': 14})
    # plt.title('L2 Convergence during L-BFGS')
    # plt.grid(True)
    plt.tight_layout()
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

    forces2 = np.array(tracker.L2_history)
    # 保存为.npy文件
    np.save('force2.npy', forces2)

    # plot figure
    # model.load_state_dict(torch.load('ritz_tmp.mdl'))
    # model.eval()
    print('load from ckpt!')
    checkpoint_path = 'ritz_3dlgb.mdl'  # put path to your trained model file if any
    output_html = "pred.html"
    # output_html1 = "pred1.html"
    save_vol_npy = True
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint:", checkpoint_path)

    with torch.no_grad():
        coords = get_draw_points().to(device)
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
    plt.show()

    isovalue = 0.0
    # build grid
    print("Building grid...")
    pts, xs, ys, zs = make_regular_grid(nx, ny, nz)
    print(f"Grid built: {nx}x{ny}x{nz} = {pts.shape[0]} points")
    # evaluate (this may take long)
    print("Evaluating model on grid (batched inference)...")
    vol = eval_model_on_grid(model, fourier, pts, nx, ny, nz, device=device, batch_size=batch_size)
    vol = gaussian_filter(vol, sigma=sigma)
    print("Evaluation done. vol.shape:", vol.shape, "min/max:", vol.min(), vol.max(), "mean:", vol.mean())
    if save_vol_npy:
        np.save(vol_npy_path, vol)
        print("Saved volume to", vol_npy_path)
    plot_volume_render(vol, xs, ys, zs, cmap='Viridis', opacity=0.6, surface_count=20, output_html=output_html)
    # plot_multilevel_isosurfaces(vol, xs, ys, zs, n_levels=15, cmap='Viridis', output_html=output_html)
    # mesh_colored_by_value(vol, xs, ys, zs, isovalue=None, smooth_sigma=0.6, output_html=output_html)

if __name__ == '__main__':
    start_time = time.time()

    # 执行你的代码

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"代码执行时间：{execution_time//60} 分,{execution_time%60:.1f}秒")
    print(f"代码执行时间：{execution_time:.2f}秒")

