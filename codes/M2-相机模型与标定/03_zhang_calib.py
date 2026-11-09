# -*- coding: utf-8 -*-
"""
M2 第 3 篇 —— 张正友标定

对应笔记：notes/M2-相机模型与标定/3-张正友标定.md

这个脚本回答五个问题
--------------------
1. 平面靶标为什么能给出单应？单应里藏着什么？
2. 从单应到内参：那两个"正交且等长"的约束是怎么来的？
3. 为什么至少要 3 个姿态？2 个会怎样？
4. 线性解出来的 K 有多准？为什么还要非线性精化？
5. 重投影误差怎么读？多少算好？

运行：
    python 03_zhang_calib.py
预期：最后打印 ALL PASS。
只依赖 numpy（LM 也手写，不调 scipy）。
"""

import numpy as np

RNG = np.random.default_rng(7)

# 真值：一台 1280x1024 的相机，8 mm 镜头
K_TRUE = np.array([[1800.0, 0.0, 640.0],
                   [0.0, 1800.0, 512.0],
                   [0.0, 0.0, 1.0]])
IMG_W, IMG_H = 1280, 1024
NOISE_PX = 0.30          # 角点检测噪声（亚像素级）


# =====================================================================
# 一、基础几何
# =====================================================================

def rodrigues(w):
    """【rodrigues】axis-angle -> R"""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def log_so3(R):
    """【log_so3】R -> axis-angle（给精化阶段当初值）"""
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(np.pi - th) < 1e-6:          # 180°，用对称部分取轴
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i] / (k[i] if k[i] > 0 else 1.0)
        return k / np.linalg.norm(k) * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * np.sin(th)) * th


def board_points(cols=7, rows=5, spacing=30.0):
    """【board_points】棋盘格内角点（世界坐标，Z=0，单位 mm）"""
    P = np.array([[c * spacing, r * spacing, 0.0]
                  for r in range(rows) for c in range(cols)], float)
    return P


def project(K, rvec, tvec, X):
    """【project】世界点 -> 像素（Z=0 的靶标也要走完整链路）"""
    Xc = X @ rodrigues(rvec).T + np.asarray(tvec, float)
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def make_views(P, n_view=8, seed=1):
    """【make_views】生成 n 组"相机看靶标"的姿态（rvec, tvec）"""
    r = np.random.default_rng(seed)
    views = []
    for _ in range(n_view):
        # 倾角 15°~35°，绕 x/y 为主，带一点绕 z
        rvec = np.array([r.uniform(-0.6, 0.6), r.uniform(-0.6, 0.6), r.uniform(-0.3, 0.3)])
        # 靶标中心大致在画面中间，距离 400~600 mm
        tvec = np.array([r.uniform(-60, 60), r.uniform(-50, 50), r.uniform(400, 600)])
        views.append((rvec, tvec))
    return views


# =====================================================================
# 二、单应：DLT + 归一化
# =====================================================================

def homography_dlt(src, dst):
    """
    【homography_dlt】DLT → 把 src(平面点) 映射到 dst(像素) 的 H

        s·[u, v, 1]ᵀ = H · [X, Y, 1]ᵀ

    归一化（Hartley）是必须的：像素坐标动辄上千，不归一化 A 的条件数会很差。
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)

    def normalize(pts):
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / (d if d > 0 else 1.0)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        return T

    Ts, Td = normalize(src), normalize(dst)
    s_h = np.hstack([src, np.ones((len(src), 1))]) @ Ts.T
    d_h = np.hstack([dst, np.ones((len(dst), 1))]) @ Td.T

    A = []
    for (x, y, _), (u, v, _) in zip(s_h, d_h):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


# =====================================================================
# 三、从单应到内参：正交 + 等长
# =====================================================================

def _v_ij(H, i, j):
    h = H
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j],
    ])


def K_from_homographies(Hs):
    """
    【K_from_homographies】Zhang 线性解 → K

    对每个 H = K[r1 r2 t]，r1、r2 满足：
        r1ᵀr2 = 0        （正交）
        |r1| = |r2| = 1  （等长）
    把 r = K⁻¹h/λ 代入，就得到两个关于 B = K⁻ᵀK⁻¹ 的线性方程。
    B 有 6 个未知数，每个姿态给 2 个方程 → 至少 3 个姿态。
    """
    V = []
    for H in Hs:
        v12 = _v_ij(H, 0, 1)
        v11 = _v_ij(H, 0, 0)
        v22 = _v_ij(H, 1, 1)
        V.append(v12)
        V.append(v11 - v22)
    V = np.asarray(V)

    _, sv, Vt = np.linalg.svd(V)
    b = Vt[-1]
    if b[0] < 0:
        b = -b
    B11, B12, B22, B13, B23, B33 = b

    denom = B11 * B22 - B12 * B12
    cy = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(lam / B11)
    fy = np.sqrt(lam * B11 / denom)
    cx = -B13 * fx * fx / lam
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, sv


def extrinsics_from_H(K, H):
    """【extrinsics_from_H】H = K[r1 r2 t] → R, t（用 SVD 强制正交）"""
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = Kinv @ H[:, 0], Kinv @ H[:, 1], Kinv @ H[:, 2]
    lam = (np.linalg.norm(h1) + np.linalg.norm(h2)) / 2
    r1, r2, t = h1 / lam, h2 / lam, h3 / lam
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


# =====================================================================
# 四、非线性精化：手写 LM
# =====================================================================

def pack(K, rvecs, tvecs):
    return np.concatenate([np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
                           np.concatenate(rvecs), np.concatenate(tvecs)])


def unpack(p, n_view):
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
    q = p[4:]
    rvecs = [q[3 * i:3 * i + 3] for i in range(n_view)]
    tt = q[3 * n_view:]
    tvecs = [tt[3 * i:3 * i + 3] for i in range(n_view)]
    return K, rvecs, tvecs


def residuals(p, P, obs, n_view):
    K, rvecs, tvecs = unpack(p, n_view)
    out = []
    for i in range(n_view):
        uv = project(K, rvecs[i], tvecs[i], P)
        out.append((uv - obs[i]).ravel())
    return np.concatenate(out)


def refine(K0, rvecs0, tvecs0, P, obs, iters=40):
    """【refine】Levenberg–Marquardt 最小化重投影误差（数值雅可比）"""
    p = pack(K0, rvecs0, tvecs0)
    n_view = len(obs)
    lam = 1e-3
    cost = np.sum(residuals(p, P, obs, n_view) ** 2)
    for _ in range(iters):
        r = residuals(p, P, obs, n_view)
        J = np.zeros((len(r), len(p)))
        h = 1e-6
        for j in range(len(p)):
            dp = np.zeros(len(p))
            dp[j] = h * max(1.0, abs(p[j]))
            J[:, j] = (residuals(p + dp, P, obs, n_view) - r) / dp[j]
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            p_new = p + delta
            cost_new = np.sum(residuals(p_new, P, obs, n_view) ** 2)
            if cost_new < cost:
                p, cost = p_new, cost_new
                lam = max(lam / 3, 1e-12)
                accepted = True
                break
            lam *= 5
        if not accepted:
            break
    K, rvecs, tvecs = unpack(p, n_view)
    rmse = np.sqrt(cost / (2 * sum(len(o) for o in obs)))
    return K, rvecs, tvecs, rmse


def rmse_of(K, rvecs, tvecs, P, obs):
    err = []
    for i, o in enumerate(obs):
        err.append((project(K, rvecs[i], tvecs[i], P) - o).ravel())
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(e ** 2))), e


# =====================================================================
# 五、demo
# =====================================================================

def build_dataset(n_view=8):
    P = board_points()
    views = make_views(P, n_view=n_view)
    obs, rvecs_true = [], []
    for rvec, tvec in views:
        uv = project(K_TRUE, rvec, tvec, P) + RNG.normal(0, NOISE_PX, (len(P), 2))
        obs.append(uv)
        rvecs_true.append(rvec)
    return P, views, obs, rvecs_true


def demo_zhang():
    print("一、合成数据")
    P, views, obs, _ = build_dataset(8)
    print("  靶标内角点: {} 个（7x5，间距 30 mm）".format(len(P)))
    print("  姿态数: {}  角点噪声: sigma={:.2f} px".format(len(obs), NOISE_PX))
    print("  真值 K: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}".format(
        K_TRUE[0, 0], K_TRUE[1, 1], K_TRUE[0, 2], K_TRUE[1, 2]))

    # 单应 + 线性 K
    Hs = [homography_dlt(P[:, :2], o) for o in obs]
    # 单应自身的重投影误差（衡量角点/求解质量）
    h_err = []
    for H, o in zip(Hs, obs):
        pr = np.hstack([P[:, :2], np.ones((len(P), 1))]) @ H.T
        pr = pr[:, :2] / pr[:, 2:3]
        h_err.append(np.sqrt(np.mean((pr - o) ** 2)))
    print("  单应重投影 RMSE: {:.3f} px".format(float(np.mean(h_err))))

    K_lin, sv = K_from_homographies(Hs)
    print("\n二、线性解（Zhang 闭式）")
    print("  fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}".format(
        K_lin[0, 0], K_lin[1, 1], K_lin[0, 2], K_lin[1, 2]))
    print("  与真值偏差: fx {:+.2f}%，fy {:+.2f}%，cx {:+.2f} px，cy {:+.2f} px".format(
        (K_lin[0, 0] / K_TRUE[0, 0] - 1) * 100,
        (K_lin[1, 1] / K_TRUE[1, 1] - 1) * 100,
        K_lin[0, 2] - K_TRUE[0, 2], K_lin[1, 2] - K_TRUE[1, 2]))
    print("  A 的最小奇异值 / 最大: {:.2e}".format(sv[-1] / sv[0]))

    # 外参初值 + 精化
    Rt = [extrinsics_from_H(K_lin, H) for H in Hs]
    rvecs0 = [log_so3(R) for R, _ in Rt]
    tvecs0 = [t for _, t in Rt]
    rmse_lin, _ = rmse_of(K_lin, rvecs0, tvecs0, P, obs)
    print("  线性解的重投影 RMSE: {:.3f} px".format(rmse_lin))

    print("\n三、LM 精化")
    K_ref, rvecs_ref, tvecs_ref, rmse_ref = refine(K_lin, rvecs0, tvecs0, P, obs)
    print("  fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}".format(
        K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2]))
    print("  与真值偏差: fx {:+.2f}%，fy {:+.2f}%，cx {:+.2f} px，cy {:+.2f} px".format(
        (K_ref[0, 0] / K_TRUE[0, 0] - 1) * 100,
        (K_ref[1, 1] / K_TRUE[1, 1] - 1) * 100,
        K_ref[0, 2] - K_TRUE[0, 2], K_ref[1, 2] - K_TRUE[1, 2]))
    print("  精化后 RMSE: {:.3f} px（噪声水平 {:.2f} px）".format(rmse_ref, NOISE_PX))

    # 每视角残差
    print("\n四、每个姿态的残差 RMSE")
    for i, o in enumerate(obs):
        e = project(K_ref, rvecs_ref[i], tvecs_ref[i], P) - o
        print("  视角 {}: {:.3f} px （点 {} 个）".format(i + 1, np.sqrt(np.mean(e ** 2)), len(o)))

    assert abs(K_ref[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.01
    assert abs(K_ref[1, 2] - K_TRUE[1, 2]) < 5.0
    assert rmse_ref < 1.5 * NOISE_PX
    return P, obs


def demo_too_few_views():
    print("\n五、姿态不够会怎样（2 个姿态）")
    P, views, obs, _ = build_dataset(2)
    Hs = [homography_dlt(P[:, :2], o) for o in obs]
    V = []
    for H in Hs:
        V.append(_v_ij(H, 0, 1))
        V.append(_v_ij(H, 0, 0) - _v_ij(H, 1, 1))
    V = np.asarray(V)
    sv = np.linalg.svd(V, compute_uv=False)
    print("  方程数 {}，未知数 6 -> rank = {}".format(V.shape[0], np.linalg.matrix_rank(V)))
    print("  最小奇异值 {:.2e}（接近 0 说明有解不唯一，K 不可信）".format(sv[-1]))
    assert np.linalg.matrix_rank(V) < 6


def main():
    demo_zhang()
    demo_too_few_views()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
