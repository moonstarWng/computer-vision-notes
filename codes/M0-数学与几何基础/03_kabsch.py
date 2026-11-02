"""
M0 第 3 篇 · SVD 求刚体变换（Kabsch）
对应笔记：notes/M0-数学与几何基础/3-SVD求刚体变换.md（§1~§6）

运行：
    python 03_kabsch.py

三个实验：正常加噪、镜像数据、共线退化。
"""

import numpy as np


def kabsch(P, Q):
    """给定一一对应的点 P, Q (N,3)，求 R,t 使 R·P + t ≈ Q"""
    p_bar, q_bar = P.mean(axis=0), Q.mean(axis=0)
    H = (P - p_bar).T @ (Q - q_bar)              # 3×3
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = float(np.sign(np.linalg.det(V @ U.T)))   # det 修正：+1 或 -1
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_bar - R @ p_bar
    return R, t, S, d


def rodrigues(w):
    """只为造测试数据用"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def residual(P, Q, R, t):
    return float(np.linalg.norm(P @ R.T + t - Q))


def demo_normal():
    print("== 1. 正常情况（30 个点，噪声 σ=1cm） ==")
    rng = np.random.default_rng(1)
    P = rng.normal(size=(30, 3))
    R_true = rodrigues(np.array([0.3, -0.5, 1.1]))
    t_true = np.array([1.0, -2.0, 0.5])
    Q = P @ R_true.T + t_true + rng.normal(0, 0.01, P.shape)

    R, t, S, d = kabsch(P, Q)
    print(f"R 误差 = {np.abs(R - R_true).max():.2e}")
    print(f"t 误差 = {np.abs(t - t_true).max():.2e}")
    print(f"奇异值 S = {S.round(3)}  det 修正 d = {d:+.0f}\n")
    return P, R_true, t_true


def demo_mirror(P, t_true):
    print("== 2. 镜像数据：不加 det 修正会得到反射矩阵 ==")
    Q = P * np.array([1.0, 1.0, -1.0]) + t_true   # 沿 z 翻一面

    R_fix, t_fix, _, d = kabsch(P, Q)
    H = (P - P.mean(0)).T @ (Q - Q.mean(0))
    U, S, Vt = np.linalg.svd(H)
    R_raw = Vt.T @ U.T                            # 不做修正
    print(f"修正后 det = {np.linalg.det(R_fix):+.1f}  残差 = {residual(P, Q, R_fix, t_fix):.3f}")
    print(f"不修正 det = {np.linalg.det(R_raw):+.1f}  残差 = {residual(P, Q, R_raw, t_fix):.3f}")
    print("→ 不修正残差更小，但它是镜像，不是旋转\n")
    assert np.isclose(np.linalg.det(R_raw), -1.0)
    assert np.isclose(np.linalg.det(R_fix), 1.0)


def demo_degenerate(R_true, t_true):
    print("== 3. 共线点：有一整个自由度不可观 ==")
    P = np.outer(np.linspace(-1, 1, 10), np.array([1.0, 0.0, 0.0]))
    Q = P @ R_true.T + t_true
    R, t, S, d = kabsch(P, Q)
    print(f"奇异值 S = {S.round(6)}")
    for th in [0.0, 0.3, 1.0, 2.5]:
        R2 = R @ rodrigues(np.array([th, 0.0, 0.0]))
        print(f"  绕 x 多转 {th:>3} rad → 残差 {residual(P, Q, R2, t):.1e}")
    print("→ 残差都一样：解不唯一，R 与真值差很多也没关系\n")


if __name__ == "__main__":
    P, R_true, t_true = demo_normal()
    demo_mirror(P, t_true)
    demo_degenerate(R_true, t_true)
    print("ALL DONE")
