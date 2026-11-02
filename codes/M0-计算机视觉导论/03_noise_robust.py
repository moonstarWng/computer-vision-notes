# -*- coding: utf-8 -*-
"""
M0 计算机视觉导论 · 第 3 篇 · 噪声与误差模型
对应笔记：notes/M0-计算机视觉导论/3-噪声与误差模型.md

四张图，把"噪声假设 → 损失函数"变具体：

    图 1  正交投影：为什么最小二乘的残差垂直于模型空间
    图 2  损失函数形状：L2 / L1 / Huber 对大残差的惩罚
    图 3  残差直方图：干净（高斯）vs 10% 离群（重尾）
    图 4  估计器对比：均值 / 中位数 / Huber 的误差

运行：
    python 03_noise_robust.py            # 直接弹窗显示
    python 03_noise_robust.py --save     # 只保存 out/noise_robust.png
"""

import os
import sys

import numpy as np

import matplotlib
SAVE = "--save" in sys.argv
if SAVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 一、正交投影
# =====================================================================

def project_onto_line(a, b):
    """
    【函数名】project = 投影；onto = 到……上；line = 直线

    把 b 投影到"由 a 张成的直线"上：p = (a·b / a·a)·a

    为什么？投影点必须落在 a 的方向上（p = t·a），
    且残差 r = b − p 必须垂直于 a（a·r = 0），解出 t 即可。
    """
    t = (a @ b) / (a @ a)
    return t * a, b - t * a


def panel_projection(ax):
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 1.0])
    p, r = project_onto_line(a, b)

    # 直线（a 的方向）
    ts = np.linspace(-0.6, 3.2, 50)
    ax.plot(ts * a[0], ts * a[1], color="#999999", linewidth=1.2,
            label="模型空间（a 的方向）")

    def arrow(v, color, label, ls="-"):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    linewidth=2, linestyle=ls))
        ax.text(v[0] * 1.05 + 0.05, v[1] * 1.05 + 0.05, label,
                color=color, fontsize=10)

    arrow(a, "#1f77b4", "a")
    arrow(b, "#2ca02c", "b（观测）")
    arrow(p, "#ff7f0e", "p（预测）")
    # 残差用虚线
    ax.annotate("", xy=b, xytext=p,
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                linewidth=2, linestyle="--"))
    ax.text((p[0] + b[0]) / 2 + 0.05, (p[1] + b[1]) / 2 + 0.05,
            "r = b − p", color="#d62728", fontsize=10)

    # 直角标记：在 p 点画一个小方块
    u = b - p
    u = u / np.linalg.norm(u)
    v = p / np.linalg.norm(p)
    s = 0.18
    ax.plot([p[0] + s * u[0], p[0] + s * u[0] + s * v[0]],
            [p[1] + s * u[1], p[1] + s * u[1] + s * v[1]],
            color="#444444", linewidth=1)
    ax.plot([p[0] + s * v[0], p[0] + s * v[0] + s * u[0]],
            [p[1] + s * v[1], p[1] + s * v[1] + s * u[1]],
            color="#444444", linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.5, 3.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("图 1  正交投影：残差垂直于模型空间\n"
                 f"a·r = {a @ r:.1f}（等于 0）", fontsize=11)


# =====================================================================
# 二、损失函数形状
# =====================================================================

def panel_loss(ax):
    r = np.linspace(-8, 8, 400)
    delta = 1.345

    l2 = 0.5 * r ** 2
    l1 = np.abs(r)
    huber = np.where(np.abs(r) <= delta,
                     0.5 * r ** 2,
                     delta * (np.abs(r) - 0.5 * delta))

    ax.plot(r, l2, label="L2（平方）", color="#d62728", linewidth=2)
    ax.plot(r, l1, label="L1（绝对值）", color="#1f77b4", linewidth=2)
    ax.plot(r, huber, label="Huber", color="#2ca02c", linewidth=2)

    # 标出 r = 5 时三者的值
    rr = 5.0
    for val, color in [(0.5 * rr ** 2, "#d62728"),
                       (abs(rr), "#1f77b4"),
                       (delta * (rr - 0.5 * delta), "#2ca02c")]:
        ax.scatter([rr], [val], color=color, zorder=5, s=30)
    ax.axvline(rr, color="#bbbbbb", linestyle=":")
    ax.text(rr + 0.2, 1.0, "离群点 r=5", fontsize=9, color="#666666")

    ax.set_ylim(0, 20)
    ax.set_xlabel("残差 r")
    ax.set_ylabel("损失 ρ(r)")
    ax.set_title("图 2  损失函数：L2 对离群点惩罚最狠", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


# =====================================================================
# 三、残差直方图
# =====================================================================

def panel_hist(ax, clean_r, dirty_r):
    bins = np.linspace(-4, 56, 120)
    ax.hist(clean_r, bins=bins, alpha=0.65, label="干净（高斯）",
            color="#1f77b4", log=True)
    ax.hist(dirty_r, bins=bins, alpha=0.65, label="10% 离群",
            color="#d62728", log=True)

    ax.axvline(0, color="#999999", linestyle=":", linewidth=1)
    ax.annotate("10 个点在这\n（重尾）",
                xy=(50, 1), xytext=(22, 3),
                fontsize=9, color="#d62728",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#d62728", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("残差 r")
    ax.set_ylabel("点数（对数）")
    ax.set_title("图 3  残差分布：干净 vs 10% 离群", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)


# =====================================================================
# 四、估计器对比
# =====================================================================

def panel_estimators(ax, errs):
    labels = list(errs["clean"].keys())
    clean = np.array([errs["clean"][k] for k in labels])
    dirty = np.array([errs["dirty"][k] for k in labels])

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, clean, w, label="干净", color="#1f77b4")
    b2 = ax.bar(x + w / 2, dirty, w, label="10% 离群", color="#d62728")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.08,
                    f"{bar.get_height():.2f}",
                    ha="center", fontsize=9)

    ax.set_xticks(x, labels)
    ax.set_ylabel("|估计 − 真值|（越接近 0 越好）")
    ax.set_ylim(0, max(dirty.max(), clean.max()) * 1.25)
    ax.set_title("图 4  三种估计器的误差\n均值被离群点拉偏，中位数/Huber 稳", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)


# =====================================================================
# 五、数据与估计器
# =====================================================================

def huber_location(x, delta=1.345, iters=100):
    """
    【函数名】huber = 胡伯（Peter Huber，人名）；location = 位置参数

    Huber + IRLS：每轮按残差给权重（大残差权重小），更新中心值。
    """
    m = float(np.median(x))
    for _ in range(iters):
        r = x - m
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        m_new = float(np.sum(w * x) / np.sum(w))
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    return m


def make_data(seed=0, mu=10.0):
    rng = np.random.default_rng(seed)
    clean = rng.normal(mu, 1.0, 100)
    dirty = clean.copy()
    dirty[:10] += 50.0          # 10% 的点错了 50
    return clean, dirty


def main():
    mu = 10.0
    clean, dirty = make_data(mu=mu)

    # 打印数字
    print("== 三种估计器（真值 10） ==")
    print(f"{'数据':<10}{'均值':>10}{'中位数':>10}{'Huber':>10}")
    for name, x in [("干净", clean), ("10% 离群", dirty)]:
        print(f"{name:<10}{x.mean():>10.3f}{np.median(x):>10.3f}"
              f"{huber_location(x):>10.3f}")

    errs = {"clean": {}, "dirty": {}}
    for name, x in [("clean", clean), ("dirty", dirty)]:
        errs[name] = {"均值": abs(x.mean() - mu),
                      "中位数": abs(np.median(x) - mu),
                      "Huber": abs(huber_location(x) - mu)}

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("噪声假设 → 损失函数：别再用 L2 硬拟合脏数据", fontsize=15)

    panel_projection(axes[0, 0])
    panel_loss(axes[0, 1])
    panel_hist(axes[1, 0], clean - mu, dirty - mu)
    panel_estimators(axes[1, 1], errs)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if SAVE:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "noise_robust.png")
        fig.savefig(out, dpi=150)
        print(f"\n已保存: {out}")

    if not SAVE:
        plt.show()
    print("ALL DONE")


if __name__ == "__main__":
    main()
