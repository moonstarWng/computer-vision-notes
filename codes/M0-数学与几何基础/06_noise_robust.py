# -*- coding: utf-8 -*-
"""
M0 第 6 篇 · 噪声与误差模型（参考篇）
对应笔记：notes/M0-数学与几何基础/6-噪声与误差模型.md

这个脚本做三件事
----------------
1. 正交投影：验证"最小二乘的残差垂直于模型空间"；
2. 三种估计器：同一批数据，均值 / 中位数 / Huber 谁更抗离群点；
3. 残差分布速查：怎么用一张表看出"重尾"。

运行：
    python 06_noise_robust.py
"""

import numpy as np


# =====================================================================
# 一、正交投影
# =====================================================================

def project_onto_line(a, b):
    """
    【函数名】project = 投影；onto = 到……上；line = 直线

    把向量 b 投影到"由 a 张成的直线"上。

    公式：
        p = (a·b / a·a) · a

    为什么是这个公式？
        投影点必须落在 a 的方向上，即 p = t·a；
        残差 r = b − p 必须垂直于 a，即 a·r = 0；
        解 t 就得到 t = (a·b)/(a·a)。

    参数：a 定义直线的方向；b 被投影的向量
    返回：投影点 p、残差 r
    """
    t = (a @ b) / (a @ a)      # @ 是点积（一维数组）
    p = t * a
    r = b - p
    return p, r


def demo_projection():
    print("== 1. 正交投影：最小二乘的几何意义 ==")
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 1.0])

    p, r = project_onto_line(a, b)
    print(f"直线方向 a = {a}")
    print(f"被投影 b   = {b}")
    print(f"投影点 p   = {p}")
    print(f"残差   r   = {r}")
    print(f"a·r = {a @ r:.1f}   ← 等于 0：残差垂直于模型空间")
    print(f"‖b‖² = {b @ b:.1f} = ‖p‖² + ‖r‖² = {p @ p:.1f} + {r @ r:.1f}"
          f"   ← 勾股定理\n")


# =====================================================================
# 二、三种估计器：均值 / 中位数 / Huber
# =====================================================================

def huber_location(x, delta=1.345, iters=100):
    """
    【函数名】huber = 胡伯（Peter Huber，人名）；location = 位置参数

    估计一组数据的"中心值"，但用 Huber 损失而不是平方损失。

    做法（IRLS，迭代重加权）：
        1) 用中位数做初值（比均值抗离群）
        2) 每一轮算残差 r = x − m
        3) 权重 w = 1（|r|≤δ）或 δ/|r|（否则）
        4) 更新 m = Σ(w·x) / Σ(w)
        5) 直到 m 不再变化

    δ 的默认 1.345 是"高斯下 95% 效率"的经验值；
    噪声尺度未知时，可以取 δ = 1.4826 · median(|r|)（MAD）。

    参数：x 一维数据；delta 阈值
    返回：鲁棒的"中心值"估计
    """
    m = float(np.median(x))                 # 初值：中位数
    for _ in range(iters):
        r = x - m
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        m_new = float(np.sum(w * x) / np.sum(w))
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    return m


def demo_estimators():
    """
    场景：真实值 μ = 10，测 100 次。
        - 干净数据：高斯噪声 σ = 1
        - 脏数据：前 10 个点额外 +50（模拟 10% 的严重误差，比如反光/粉尘）

    三个估计器：
        均值   ← 等价于平方损失（L2）
        中位数 ← 等价于绝对值损失（L1）
        Huber  ← 近高斯 + 重尾
    """
    print("== 2. 三种估计器：谁更抗离群点 ==")
    rng = np.random.default_rng(0)
    mu = 10.0

    clean = rng.normal(mu, 1.0, 100)
    dirty = clean.copy()
    dirty[:10] += 50.0                      # 10% 的点错了 50

    print(f"{'数据':<10}{'真值':>6}{'均值':>10}{'中位数':>10}{'Huber':>10}")
    for name, x in [("干净", clean), ("10% 离群", dirty)]:
        print(f"{name:<10}{mu:>6.1f}{x.mean():>10.3f}"
              f"{np.median(x):>10.3f}{huber_location(x):>10.3f}")

    print("\n误差（越接近 0 越好）：")
    print(f"{'数据':<10}{'|均值−真值|':>14}{'|中位数−真值|':>16}{'|Huber−真值|':>14}")
    for name, x in [("干净", clean), ("10% 离群", dirty)]:
        print(f"{name:<10}{abs(x.mean() - mu):>14.3f}"
              f"{abs(np.median(x) - mu):>16.3f}"
              f"{abs(huber_location(x) - mu):>14.3f}")
    print("→ 干净时三者都对；有离群点时，均值被拉偏约 5，另外两个几乎不动\n")


# =====================================================================
# 三、残差分布速查：怎么看出"重尾"
# =====================================================================

def residual_summary(x, mu=10.0):
    """
    【函数名】residual = 残差；summary = 汇总

    把残差按绝对值分段计数。高斯数据的计数会快速衰减；
    重尾数据的"远处"会突然多出一撮点 —— 这就是重尾的直接证据。
    """
    r = np.abs(x - mu)
    edges = [0, 1, 2, 3, 5, 10, np.inf]
    counts = np.histogram(r, bins=edges)[0]
    return counts


def demo_residuals():
    print("== 3. 残差分布速查（|r| 分段的点数） ==")
    rng = np.random.default_rng(1)
    mu = 10.0
    clean = rng.normal(mu, 1.0, 100)
    dirty = clean.copy()
    dirty[:10] += 50.0

    print(f"{'|r| 范围':<14}{'干净':>8}{'10% 离群':>10}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, np.inf)]:
        label = f"[{lo}, {hi})" if np.isfinite(hi) else f"[{lo}, ∞)"
        c1 = np.sum((np.abs(clean - mu) >= lo) & (np.abs(clean - mu) < hi))
        c2 = np.sum((np.abs(dirty - mu) >= lo) & (np.abs(dirty - mu) < hi))
        print(f"{label:<14}{c1:>8}{c2:>10}")
    print("→ 干净数据在 5σ 之外是 0；离群数据在 10σ 之外还有 10 个")
    print("→ 这张表就是'重尾'的体检报告：远处不该有点，却有点\n")


# =====================================================================
# 四、主流程
# =====================================================================

def main():
    demo_projection()
    demo_estimators()
    demo_residuals()

    # 一个断言：干净数据下，三种估计器都不该差太远
    rng = np.random.default_rng(2)
    x = rng.normal(10.0, 1.0, 200)
    vals = [x.mean(), np.median(x), huber_location(x)]
    assert max(vals) - min(vals) < 0.5, "干净数据下三种估计器应该很接近"
    print("ALL DONE")


if __name__ == "__main__":
    main()
