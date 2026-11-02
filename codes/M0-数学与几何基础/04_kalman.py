"""
M0 第 4 篇 · 协方差传播、马氏距离、卡尔曼
对应笔记：notes/M0-数学与几何基础/4-协方差传播与卡尔曼.md（§2~§6）

运行：
    python 04_kalman.py

三个实验：一阶误差传播 vs 蒙特卡洛、两个高斯融合、常速度模型 KF。
"""

import numpy as np


# ---------------------------------------------------------------- 误差传播
def f(v):
    """非线性函数：y1 = x1x2, y2 = x1/x2"""
    return np.array([v[0] * v[1], v[0] / v[1]])


def demo_propagation():
    print("== 1. 误差传播：一阶泰勒 vs 蒙特卡洛 ==")
    mu = np.array([2.0, 4.0])
    Sig = np.array([[0.04, 0.01],
                    [0.01, 0.09]])

    J = np.array([[mu[1], mu[0]],
                  [1.0 / mu[1], -mu[0] / mu[1] ** 2]])
    Sig_y = J @ Sig @ J.T                     # Σ_y ≈ J Σ_x J^T

    rng = np.random.default_rng(2)
    samples = rng.multivariate_normal(mu, Sig, 200_000)
    fy = np.array([f(s) for s in samples])
    Sig_mc = np.cov(fy.T)

    print("解析传播 Σ_y =\n", Sig_y.round(5))
    print("蒙特卡洛 Σ_y =\n", Sig_mc.round(5))
    print(f"相对误差 = {np.abs(Sig_y - Sig_mc).max() / np.abs(Sig_mc).max():.1%}\n")


# ---------------------------------------------------------------- 两个高斯融合
def fuse(m1, s1, m2, s2):
    """一维：先验 N(m1,s1) 与观测 N(m2,s2) 的加权融合"""
    K = s1 / (s1 + s2)                 # 一维时 K = P^-/(P^-+R)
    return m1 + K * (m2 - m1), (1 - K) * s1, K


def demo_fuse():
    print("== 2. 两个高斯融合：K 就是观测的话语权 ==")
    m, p, K = fuse(0.0, 1.0, 2.0, 1.0)
    print(f"先验 N(0,1) + 观测 z=2 (R=1) → x={m:.3f}, P={p:.3f}, K={K:.3f}\n")

    print(f"{'R':>6}{'K':>9}{'x':>9}{'P':>9}")
    for R in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m, p, K = fuse(0.0, 1.0, 2.0, R)
        print(f"{R:>6}{K:>9.4f}{m:>9.3f}{p:>9.3f}")
    print("→ R 越小（观测越准），K 越接近 1\n")


# ---------------------------------------------------------------- 卡尔曼
def demo_kalman():
    print("== 3. 常速度模型 KF（只测位置，速度靠模型推） ==")
    F = np.array([[1.0, 1.0], [0.0, 1.0]])    # 位置 += 速度
    Q = np.diag([0.01, 0.04])                 # 模型噪声
    H = np.array([[1.0, 0.0]])                # 只观测位置
    R = np.array([[0.5]])                     # 观测噪声

    x = np.array([0.0, 0.0])                  # [位置, 速度]
    P = np.eye(2)
    print(f"{'z':>6}{'x_pos':>9}{'x_vel':>9}{'K0':>9}")
    for z in [1.0, 2.1, 2.9, 4.2, 5.1]:
        x = F @ x                             # 预测
        P = F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + (K @ (np.array([[z]]) - H @ x)).ravel()   # 更新
        P = (np.eye(2) - K @ H) @ P
        print(f"{z:>6.1f}{x[0]:>9.3f}{x[1]:>9.3f}{K[0, 0]:>9.3f}")
    print(f"\n最终 x = {x.round(3)}  （真值应该是位置 5、速度 1 附近）")
    print(f"最终 P =\n{P.round(4)}\n")


def demo_mahalanobis():
    print("== 4. 马氏距离：按不确定度归一化 ==")
    e = np.array([0.10, 0.0])                       # 偏差 10 cm
    for name, sigma in [("精确传感器 σ=1mm", 0.001), ("普通传感器 σ=20cm", 0.20)]:
        S = np.diag([sigma ** 2, sigma ** 2])
        d = float(np.sqrt(e @ np.linalg.inv(S) @ e))
        print(f"{name:<18} → 马氏距离 d = {d:6.2f}")
    print("（约 95% 置信门限：2 维卡方 5.99 → d≈2.45；超过就该怀疑是异常）")


if __name__ == "__main__":
    demo_propagation()
    demo_fuse()
    demo_kalman()
    demo_mahalanobis()
    print("\nALL DONE")
