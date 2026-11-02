# -*- coding: utf-8 -*-
"""
M0 第 3 篇 · SVD 求刚体变换（Kabsch）
对应笔记：notes/M0-数学与几何基础/3-SVD求刚体变换.md（§1~§6）

这个脚本回答四个问题
--------------------
1. 为什么平移 t 可以"一步写出来"，不用优化？
2. 为什么求 R 最后变成"对 3×3 矩阵做一次 SVD"？
3. 为什么最后要乘 diag(1,1,det(VU^T))？不乘会出什么？
4. 点云共线时，为什么残差为 0 但解是错的？

运行
----
    python 03_kabsch.py

三个实验：正常加噪、镜像数据、共线退化。
"""

import numpy as np


# =====================================================================
# 一、主角：Kabsch 算法
# =====================================================================

def kabsch(P, Q):
    """
    给定一一对应的两组点 P、Q（都是 N×3，一行一个点），
    求旋转 R 和平移 t，使 R·p_i + t 尽量接近 q_i。

    目标函数：
        min  Σ ‖R·p_i + t − q_i‖²

    推导分三步（笔记 §1~§3）：

    第 1 步：对 t 求导令 0
        t = q̄ − R·p̄
        其中 p̄、q̄ 是两组点的质心。也就是说，最优平移就是
        "把旋转后的源质心搬到目标质心"。

    第 2 步：把 t 代回去，用去质心坐标 p̃ = p − p̄、q̃ = q − q̄
        Σ‖R·p̃ − q̃‖² = 常数 − 2·tr(R·H),    H = Σ p̃·q̃^T
        于是"最小化残差"变成"最大化 tr(R·H)"。

    第 3 步：对 H 做 SVD，H = U·S·V^T
        tr(R·H) = tr(R·U·S·V^T) = tr(V^T·R·U·S) ≤ Σ s_i
        取等号当且仅当 V^T·R·U = I，即 R = V·U^T。

    但 V·U^T 可能是"反射"（det = -1），所以要做 det 修正：
        d = sign(det(V·U^T))
        R = V · diag(1, 1, d) · U^T

    参数：P, Q，形状 (N,3)，必须一一对应
    返回：R (3×3)、t (3,)、奇异值 S (3,)、det 修正量 d
    """
    # ---- 第 1 步：质心（均值），以及去质心坐标 ----
    p_bar, q_bar = P.mean(axis=0), Q.mean(axis=0)   # axis=0 表示"按列求平均"
    Pc = P - p_bar                                  # 每行减去质心
    Qc = Q - q_bar

    # ---- 第 2 步：3×3 的相关矩阵 ----
    # 注意：如果点按"行"排列，H = Pc^T @ Qc；按"列"排列则是 Pc @ Qc^T。
    # 这两种写法差一个转置，公式里的 U、V 也会跟着换位，别混。
    H = Pc.T @ Qc

    # ---- 第 3 步：SVD ----
    # numpy 返回的第三个是 V^T（不是 V），这是最容易写错的地方
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # det 修正：d = ±1，保证最后 det(R) = +1
    d = float(np.sign(np.linalg.det(V @ U.T)))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T

    # ---- 平移：由 t = q̄ − R·p̄ 直接得到 ----
    t = q_bar - R @ p_bar
    return R, t, S, d


def rodrigues(w):
    """
    旋转向量 -> 旋转矩阵（这里只用来造测试数据）

    公式见第 1 篇；实现几乎一样，保留一份是为了让本文件可以独立运行。
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def residual(P, Q, R, t):
    """
    整体残差：‖R·p_i + t − q_i‖ 的总和（Frobenius 范数）

    注意 P @ R.T 是"对每一行做 R·p"的写法：
        (R @ p)^T = p^T @ R^T
    这样比循环快很多。
    """
    return float(np.linalg.norm(P @ R.T + t - Q))


# =====================================================================
# 二、三个实验
# =====================================================================

def demo_normal():
    """
    正常情况：30 个点，加 1cm 噪声。

    看两件事：
        1) 解出的 R、t 和真值差多少（误差应该和噪声同量级）；
        2) 奇异值 S 是不是三个都明显大于 0
           （都大于 0 说明点云"张得开"，三个方向都有信息）。
    """
    print("== 1. 正常情况（30 个点，噪声 σ=1cm） ==")
    rng = np.random.default_rng(1)

    P = rng.normal(size=(30, 3))                    # 源点云
    R_true = rodrigues(np.array([0.3, -0.5, 1.1]))  # 真实旋转
    t_true = np.array([1.0, -2.0, 0.5])             # 真实平移
    # 目标点云 = 旋转 + 平移 + 噪声
    Q = P @ R_true.T + t_true + rng.normal(0, 0.01, P.shape)

    R, t, S, d = kabsch(P, Q)
    print(f"R 误差 = {np.abs(R - R_true).max():.2e}")
    print(f"t 误差 = {np.abs(t - t_true).max():.2e}")
    print(f"奇异值 S = {S.round(3)}  det 修正 d = {d:+.0f}\n")
    return P, R_true, t_true


def demo_mirror(P, t_true):
    """
    镜像数据：目标点云是源点云"翻了一面"。

    这时"完美拟合"的解是一个反射矩阵（det = -1），不是旋转。
    对比两行输出：
        修正后：det=+1，残差变大 —— 但它是合法的旋转；
        不修正：det=-1，残差更小 —— 但物理上做不到（等于照镜子）。

    这就是题库 Q1 的坑：残差小不等于解对。
    """
    print("== 2. 镜像数据：不加 det 修正会得到反射矩阵 ==")
    Q = P * np.array([1.0, 1.0, -1.0]) + t_true     # 沿 z 轴翻一面

    R_fix, t_fix, _, d = kabsch(P, Q)               # 带 det 修正

    # 手动做一遍"不带修正"的 SVD，用来对比
    H = (P - P.mean(0)).T @ (Q - Q.mean(0))
    U, S, Vt = np.linalg.svd(H)
    R_raw = Vt.T @ U.T                              # 省略 diag(1,1,d)

    print(f"修正后 det = {np.linalg.det(R_fix):+.1f}  残差 = {residual(P, Q, R_fix, t_fix):.3f}")
    print(f"不修正 det = {np.linalg.det(R_raw):+.1f}  残差 = {residual(P, Q, R_raw, t_fix):.3f}")
    print("→ 不修正残差更小，但它是镜像，不是旋转\n")

    # 断言：这两个 det 必须是 +1 和 -1，否则说明镜像数据造错了
    assert np.isclose(np.linalg.det(R_raw), -1.0)
    assert np.isclose(np.linalg.det(R_fix), 1.0)


def demo_degenerate(R_true, t_true):
    """
    共线退化：10 个点全部落在一条直线上（x 轴）。

    此时 H 的秩只有 1，S = [s, 0, 0]：
        - 绕这条直线的旋转完全不影响残差 -> 这个自由度根本不可观；
        - 所以解不唯一，Kabsch 只是"随便"给了其中一个。

    这正是"残差为 0 但解是错的"的最干净例子。
    工程上要检查 s2/s1、s3/s1 是否接近 0。
    """
    print("== 3. 共线点：有一整个自由度不可观 ==")
    # 外积：一个列向量 × 一个行向量，得到 10 个点都落在 x 轴上
    P = np.outer(np.linspace(-1, 1, 10), np.array([1.0, 0.0, 0.0]))
    Q = P @ R_true.T + t_true

    R, t, S, d = kabsch(P, Q)
    print(f"奇异值 S = {S.round(6)}")               # 后两个应该是 0

    # 在解的基础上，再绕 x 轴额外转任意角度，残差都不变
    for th in [0.0, 0.3, 1.0, 2.5]:
        R2 = R @ rodrigues(np.array([th, 0.0, 0.0]))
        print(f"  绕 x 多转 {th:>3} rad → 残差 {residual(P, Q, R2, t):.1e}")
    print("→ 残差都一样：解不唯一，R 与真值差很多也没关系\n")


if __name__ == "__main__":
    P, R_true, t_true = demo_normal()
    demo_mirror(P, t_true)
    demo_degenerate(R_true, t_true)
    print("ALL DONE")
