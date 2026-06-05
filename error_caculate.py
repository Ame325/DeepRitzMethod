import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from scipy.integrate import trapezoid

# 设置 MathText 字体集为 "cm"（Computer Modern）或 "stix"
plt.rcParams['mathtext.fontset'] = 'cm'   # 'cm' 最接近 LaTeX 默认数学字体

def periodic_shift_1d(arr, shift):
    """一维周期平移（整数网格点）"""
    return np.roll(arr, shift)


def compare_periodic_solutions_1d(u_ref, u_pred, x=None, plot=True):
    """
    比较两个一维周期解，自动寻找最佳平移并对齐。

    Parameters
    ----------
    u_ref : ndarray, shape (N,)
        参考解
    u_pred : ndarray, shape (N,)
        待比较解
    x : ndarray or None
        网格坐标；若 None 则用 np.arange(N)
    plot : bool
        是否分开画图

    Returns
    -------
    result : dict
    """
    u_ref = np.asarray(u_ref).reshape(-1)
    u_pred = np.asarray(u_pred).reshape(-1)

    if u_ref.shape != u_pred.shape:
        raise ValueError(f"u_ref.shape={u_ref.shape}, u_pred.shape={u_pred.shape} 不一致")

    N = len(u_ref)

    if x is None:
        x = np.arange(N)
    else:
        x = np.asarray(x).reshape(-1)
        if len(x) != N:
            raise ValueError(f"x长度={len(x)} 与数据长度N={N} 不一致")

    # 如果是 [0,1] 的周期网格，建议你传入 endpoint=False 的 x
    # 这里不强制，但如果你传了 endpoint=True，最后一个点可能和第一个点重复

    # --------------------------
    # 直接暴力搜索最佳周期平移
    # --------------------------
    best_shift = 0
    best_err = np.inf
    best_aligned = None

    for s in range(N):
        cand = periodic_shift_1d(u_pred, s)
        err = np.linalg.norm(u_ref - cand)
        if err < best_err:
            best_err = err
            best_shift = s
            best_aligned = cand

    # 也检查负方向对应的等价表示，取更好的
    # 这样更符合“向左/向右平移”的直觉
    if best_shift > N // 2:
        best_shift = best_shift - N

    cand2 = periodic_shift_1d(u_pred, -best_shift)
    err2 = np.linalg.norm(u_ref - cand2)

    if err2 < best_err:
        best_err = err2
        best_aligned = cand2
        best_shift = -best_shift

    abs_err = np.linalg.norm(u_ref - best_aligned)
    rel_err = abs_err / np.linalg.norm(u_ref)

    abs_err2 = np.sqrt(((u_ref - best_aligned)**2).mean())

    t1 = (u_ref - best_aligned)**2
    x = np.linspace(0, 1, t1.shape[0])
    dx = x[1] - x[0]
    result = trapezoid(t1, dx=dx)
    abs_test = np.sqrt(result)

    print("=" * 50)
    print(f"Best periodic shift = {best_shift}")
    print(f"Absolute L2 error   = {abs_err:.6e}")
    print(f"Absolute L2 error2   = {abs_err2:.6e}")
    print(f"Relative L2 error   = {rel_err:.6e}")
    print(f"Absolute L_infinity error = {np.max(abs(u_ref-best_aligned)):.6e}")
    print(f"test error = {abs_test:.6e}")
    print("=" * 50)

    if plot:
        # =========================
        # Comparison
        # =========================
        plt.figure(figsize=(8, 4))
        plt.plot(x, best_aligned, label='DL', color='red', lw=3)
        plt.plot(x, u_ref , '--', label='FDM')

        # plt.title('pred')
        plt.ylim(-1.1, 1.1)
        # 获取当前坐标轴
        ax = plt.gca()
        # 显示图例
        plt.legend()

        plt.xlabel('x')
        plt.ylabel('u(x)')
        # plt.grid(True)
        plt.show()

        # =========================
        # Comparison error
        # =========================
        plt.figure(figsize=(8, 4))
        plt.plot(x, abs(best_aligned - u_ref))

        # plt.title('pred')
        # plt.ylim(-1.1, 1.1)

        plt.xlabel('x')
        plt.ylabel('$|\phi_\mathrm{FDM}-\phi_\mathrm{DL}|$', fontsize=16)
        plt.xlabel('$x$', fontsize=16)
        plt.tight_layout()
        # plt.grid(True)
        plt.show()


        # =========================
        # Reference
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(x, u_ref)

        plt.title("Reference")
        plt.xlabel("x")
        plt.ylabel("u")

        plt.tight_layout()
        plt.show()


        # =========================
        # Original
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(x, u_pred)

        plt.title("Original")
        plt.xlabel("x")
        plt.ylabel("u")

        plt.tight_layout()
        plt.show()


        # =========================
        # Aligned
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(x, best_aligned)

        plt.title("Aligned")
        plt.xlabel("x")
        plt.ylabel("u")

        plt.tight_layout()
        plt.show()


        # =========================
        # Absolute Error
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(x, np.abs(u_ref - best_aligned))

        plt.title("Absolute Error")
        plt.xlabel("x")
        plt.ylabel("error")

        plt.tight_layout()
        plt.show()

    return {
        "shift": best_shift,
        "relative_l2_error": rel_err,
        "absolute_l2_error": abs_err,
        "absolute_l2_error2": abs_err2,
        "aligned_solution": best_aligned
    }

def periodic_shift(arr, shift_x, shift_y):
    """
    周期平移（整数网格点）

    Parameters
    ----------
    arr : ndarray, shape (Nx, Ny)
    shift_x : int
        x方向平移（axis=0）
    shift_y : int
        y方向平移（axis=1）

    Returns
    -------
    shifted : ndarray
    """
    return np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)


def find_best_periodic_shift(u_ref, u_pred):
    """
    使用 FFT 相位相关寻找最佳周期平移

    Parameters
    ----------
    u_ref : ndarray
        参考解
    u_pred : ndarray
        待对齐解

    Returns
    -------
    best_shift : tuple
        (shift_x, shift_y)
    """

    # 去均值，避免常数项影响
    a = u_ref - np.mean(u_ref)
    b = u_pred - np.mean(u_pred)

    # FFT
    Fa = np.fft.fftn(a)
    Fb = np.fft.fftn(b)

    # Phase correlation
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-14)

    corr = np.fft.ifftn(R).real

    # 峰值位置
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    shift = np.array(peak, dtype=int)

    # 转成有符号平移
    for i, n in enumerate(u_ref.shape):
        if shift[i] > n // 2:
            shift[i] -= n

    return tuple(shift)


def relative_l2_error(u, v):
    """
    相对 L2 误差
    """
    return np.linalg.norm(u - v) / np.linalg.norm(u)


def absolute_l2_error(u, v):
    """
    绝对 L2 误差
    """
    return np.linalg.norm(u - v)


def compare_periodic_solutions(u_ref, u_pred, plot=True):
    """
    比较两个周期解：
    自动寻找最佳平移并计算误差

    Parameters
    ----------
    u_ref : ndarray
    u_pred : ndarray
    plot : bool

    Returns
    -------
    result : dict
    """

    # 找最佳平移
    sx, sy = find_best_periodic_shift(u_ref, u_pred)

    # 两个方向都试一下
    cand1 = periodic_shift(u_pred, sx, sy)
    cand2 = periodic_shift(u_pred, -sx, -sy)

    err1 = relative_l2_error(u_ref, cand1)
    err2 = relative_l2_error(u_ref, cand2)

    if err1 <= err2:
        best = cand1
        best_shift = (sx, sy)
        rel_err = err1
    else:
        best = cand2
        best_shift = (-sx, -sy)
        rel_err = err2

    abs_err = absolute_l2_error(u_ref, best)

    abs_err2 = np.sqrt(((u_ref.reshape(-1) - best.reshape(-1))**2).mean())

    t1 = (u_ref - best)**2
    x = np.linspace(0, 1, t1.shape[0])
    y = np.linspace(0, 1, t1.shape[1])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')   # 注意 indexing='xy' 得到形状 (ny, nx)
    # 二重积分
    int_x = trapezoid(t1, dx=dx, axis=1)   # 先对 x 积分
    rs = trapezoid(int_x, dx=dy)      # 再对 y 积分
    print(rs)
    abs_test = np.sqrt(rs)

    print("=" * 50)
    print("Best periodic shift:")
    print(f"shift_x = {best_shift[0]}")
    print(f"shift_y = {best_shift[1]}")
    print("-" * 50)
    print(f"Relative L2 error = {rel_err:.6e}")
    print(f"Absolute L2 error = {abs_err:.6e}")
    print(f"Absolute L2 error2 = {abs_err2:.6e}")
    print(f"Absolute L_infinity error = {np.max(abs(u_ref-best)):.6e}")
    print(f"test error = {abs_test:.6e}")
    print("=" * 50)

    if plot:
        # =========================
        # Reference
        # =========================
        plt.figure(figsize=(5, 5))

        im0 = plt.imshow(
            u_ref,
            origin='lower',
            cmap='viridis'
        )

        plt.title("Reference")
        plt.colorbar(im0)
        plt.tight_layout()
        plt.show()

        # =========================
        # Original
        # =========================
        plt.figure(figsize=(5, 5))

        im1 = plt.imshow(
            u_pred,
            origin='lower',
            cmap='viridis'
        )

        plt.title("Original")
        plt.colorbar(im1)
        plt.tight_layout()
        plt.show()

        # =========================
        # Aligned
        # =========================
        plt.figure(figsize=(5, 5))

        im2 = plt.imshow(
            best,
            origin='lower',
            cmap='viridis'
        )

        plt.title("Aligned")
        plt.colorbar(im2)
        plt.tight_layout()
        plt.show()

        # =========================
        # Absolute Error
        # =========================
        # plt.figure(figsize=(10, 8))
        #
        # im3 = plt.imshow(
        #     np.abs(u_ref - best),
        #     origin='lower',
        #     cmap='hot',
        #     extent=[0, 1, 0, 1],
        # )
        #
        # plt.title("Absolute Error")
        # plt.colorbar(im3)
        # plt.tight_layout()
        # plt.show()
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1)
        h = plt.imshow(np.abs(u_ref - best), interpolation='nearest', cmap='hot',
                       extent=[0, 1, 0, 1],
                       origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax)
        plt.show()

    return {
        "shift": best_shift,
        "relative_l2_error": rel_err,
        "absolute_l2_error": abs_err,

        "aligned_solution": best
    }
#######################################################二维：
dim = 1
if dim == 1:
    u1 = np.load('y_sup.npy')
    u2 = np.load('dim1tmp1.npy')
    # u2 = u2[:-1]
    x = np.linspace(0, 1, len(u1), endpoint=True)

    result = compare_periodic_solutions_1d(
        u1,
        u2,
        x=x,
        plot=True
    )

if dim == 2:
    # data = loadmat("circle_state_N_200_FDM.mat")
    # u1 = data["u_vec"]
    # data = loadmat("case11.mat")
    # u2 = data["data"].T

    # u2 = np.load('lamella202.npy')
    # u1 = np.load('lamella_ML_trapz.npy')

    u2 = np.load('circle_202_case2.npy')
    u1 = np.load('circle2_trapz.npy').T


    result = compare_periodic_solutions(u1, u2)

    print(result["shift"])
    print(result["relative_l2_error"])