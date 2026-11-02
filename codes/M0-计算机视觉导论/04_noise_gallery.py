# -*- coding: utf-8 -*-
"""
M0 计算机视觉导论 · 造噪声给你看
对应笔记：notes/M0-计算机视觉导论/3-噪声与误差模型.md

两张图：
    图 A  一张"零件"图像，分别加 8 种噪声，看它们长什么样
    图 B  一片点云，加"粉尘"和"反光鬼影"，看 3D 里的噪声

运行：
    python 04_noise_gallery.py            # 弹窗
    python 04_noise_gallery.py --save     # 存到 out/noise_gallery.png / out/pointcloud_noise.png
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

rng = np.random.default_rng(0)


# =====================================================================
# 一、先造一张"干净"的零件图
# =====================================================================

def make_scene(h=240, w=320):
    """
    合成一张灰度图：
        - 背景：从左到右的渐变（模拟光照不均）
        - 中间：一块亮的长方形（零件）
        - 两个圆孔（暗）
        - 一条斜条纹（纹理）
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(float)

    img = 55 + 140 * (xx / w)                     # 背景渐变（拉大，方便看量化）

    # 零件主体（亮）
    body = (np.abs(xx - w * 0.50) < w * 0.28) & (np.abs(yy - h * 0.50) < h * 0.26)
    img[body] = 205

    # 两个圆孔（暗）
    for cx, cy, r in [(w * 0.38, h * 0.42, 18), (w * 0.62, h * 0.58, 14)]:
        hole = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        img[hole] = 120

    # 斜条纹
    stripe = np.abs((xx - 0.55 * yy) % 40 - 20) < 3
    img[stripe & body] = 175

    return np.clip(img, 0, 255)


def u8(x):
    """把浮点图裁到 [0,255] 再转成 8 位，模拟相机输出"""
    return np.clip(x, 0, 255).astype(np.uint8)


# =====================================================================
# 二、八种噪声
# =====================================================================

def add_gaussian(img, sigma=12):
    """加性高斯：热噪声/读出噪声，每个像素独立加 N(0, σ²)"""
    return img + rng.normal(0, sigma, img.shape)


def add_salt_pepper(img, ratio=0.02):
    """椒盐/脉冲：随机像素变成 0 或 255（死像素、传输错误）"""
    out = img.copy()
    n = int(ratio * img.size)
    idx = rng.choice(img.size, n, replace=False)
    vals = rng.choice([0.0, 255.0], n)
    out.flat[idx] = vals
    return out


def add_shot(img, scale=8):
    """
    散粒噪声：光子计数服从泊松分布，方差 = 均值。
    光子越少（暗处）越明显 —— 这是它和高斯噪声最大的区别。
    """
    return rng.poisson(np.maximum(img, 0) / scale) * scale


def add_speckle(img, strength=0.35):
    """
    散斑：激光/超声相干成像的乘性噪声。
    I_noisy = I · (1 + n)，亮的区域噪声也大。
    """
    return img * (1 + rng.normal(0, strength, img.shape))


def quantize(img, levels=8):
    """量化：A/D 位数太少，灰度被分成几档"""
    step = 256 / levels
    return np.floor(img / step) * step


def add_glare(img, cx=None, cy=None, gain=220, sigma=28):
    """
    强反光：局部过曝，像素被截断到 255（饱和），信息永久丢失。
    这不是"随机噪声"，是系统性偏差 —— 平均多少次都回不来。
    """
    h, w = img.shape
    cx = w * 0.62 if cx is None else cx
    cy = h * 0.35 if cy is None else cy
    yy, xx = np.mgrid[0:h, 0:w]
    blob = gain * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return img + blob


def add_dust(img, n=250):
    """
    粉尘：空中颗粒散射，图像上是随机亮斑（点云里是悬浮的孤立点）。
    """
    out = img.copy()
    h, w = img.shape
    for _ in range(n):
        y = rng.integers(2, h - 2)
        x = rng.integers(2, w - 2)
        out[y - 1:y + 2, x - 1:x + 2] = np.minimum(out[y - 1:y + 2, x - 1:x + 2] + 90, 255)
    return out


# =====================================================================
# 三、图 A：噪声画廊
# =====================================================================

def figure_images():
    clean = make_scene()

    variants = [
        ("干净", clean),
        ("高斯 σ=12\n（热噪声）", add_gaussian(clean)),
        ("椒盐 2%\n（死像素）", add_salt_pepper(clean)),
        ("散粒（泊松）\n方差∝亮度", add_shot(clean)),
        ("散斑（乘性）\nρ=0.35", add_speckle(clean)),
        ("量化 3 bit\n（8 级灰度）", quantize(clean)),
        ("强反光\n（过曝饱和）", add_glare(clean)),
        ("粉尘 + 高斯\n（产线综合）", add_dust(add_gaussian(clean, 6))),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8))
    fig.suptitle("八种噪声：同一张零件图，不同的坏法", fontsize=15)

    for ax, (title, img) in zip(axes.ravel(), variants):
        ax.imshow(u8(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03,
                        wspace=0.06, hspace=0.28)
    return fig


# =====================================================================
# 四、图 B：点云里的噪声
# =====================================================================

def make_cloud(n=45):
    """一片带轻微起伏的平面点云"""
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.04 * np.sin(2.5 * X) * np.cos(2.5 * Y)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def figure_cloud():
    clean = make_cloud()

    surface = clean + rng.normal(0, 0.004, clean.shape)       # 测距噪声
    dust = np.column_stack([rng.uniform(-1, 1, 180),           # 粉尘：空间里随机悬浮
                            rng.uniform(-1, 1, 180),
                            rng.uniform(-0.05, 0.25, 180)])
    ghost = clean[rng.choice(len(clean), 60, replace=False)]   # 反光鬼影：整片平移
    ghost = ghost - np.array([0, 0, 0.18]) + rng.normal(0, 0.01, (60, 3))

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("点云里的两种“脏”：粉尘（悬浮点）+ 反光鬼影（虚假面）", fontsize=14)

    for i, (title, pts, color) in enumerate([
            ("干净点云", clean, "#888888"),
            ("噪声点云", surface, "#888888")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=color, label="表面")
        if i == 1:
            ax.scatter(dust[:, 0], dust[:, 1], dust[:, 2], s=12, c="#d62728", label="粉尘")
            ax.scatter(ghost[:, 0], ghost[:, 1], ghost[:, 2], s=10, c="#ff7f0e", label="鬼影")
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_zlim(-0.3, 0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-60)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# =====================================================================
# 五、主流程
# =====================================================================

def main():
    fig1 = figure_images()
    fig2 = figure_cloud()

    if SAVE:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in [("noise_gallery.png", fig1), ("pointcloud_noise.png", fig2)]:
            p = os.path.join(out_dir, name)
            fig.savefig(p, dpi=150)
            print("saved:", p)

    if not SAVE:
        plt.show()
    print("ALL DONE")


if __name__ == "__main__":
    main()
