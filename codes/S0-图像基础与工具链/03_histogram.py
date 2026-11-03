# -*- coding: utf-8 -*-
"""
S0 第 3 篇 —— 直方图与点运算：均衡、伽马、阈值

对应笔记：notes/S0-图像基础与工具链/3-直方图与点运算.md

这个脚本回答五个问题
--------------------
1. 直方图到底统计什么？为什么要先看它再调阈值？
2. 直方图均衡化为什么能把"挤在中间"的灰度拉开？公式怎么推？
3. 均衡化会放大噪声，CLAHE 是怎么补救的？
4. 伽马校正在改什么？为什么显示器和相机都要它？
5. Otsu 阈值为什么"自动"？它在什么情况下会失效？

运行：
    python 03_histogram.py           # 弹窗
    python 03_histogram.py --save    # 存到 out/histogram.png
预期：最后打印 ALL PASS。
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
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# =====================================================================
# 一、直方图：先看数据分布，再决定怎么处理
# =====================================================================

def histogram(img, bins=256, value_range=(0, 255)):
    """
    【histogram】histogram → 直方图

    hist[k] = 灰度值落在第 k 个 bin 的像素个数。

    用 np.bincount 而不是 Python 循环：
        先把像素值转成整数下标，再数每个下标出现几次。
    归一化后 hist / N 就是"灰度值的经验概率分布" p(k)。
    """
    img = np.asarray(img)
    idx = np.clip(img.astype(np.int64), value_range[0], value_range[1] - 1)
    hist = np.bincount(idx.ravel(), minlength=bins)[:bins]
    return hist.astype(np.float64)


def entropy(img):
    """
    【entropy】entropy → 信息熵

        H = -Σ p(k) log2 p(k)    单位：bit

    熵衡量"灰度有多少种可能性"：
        全黑图只有 1 个值 -> H = 0
        均匀分布 256 级 -> H = 8 bit
    均衡化的目标就是让 p(k) 尽量均匀（熵尽量大）。
    """
    p = histogram(img) / img.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def make_low_contrast(h=160, w=220):
    """
    【make_low_contrast】low contrast scene → 低对比度图

    模拟"曝光不足 + 背光"：有用的灰度全挤在 100~140，
    背景占了大半，真正的细节只差几个灰阶。
    直方图会表现为一个很窄的峰。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # 背景：缓慢的亮度起伏（100~118）
    img = 108 + 10 * np.sin(xx / 35.0) * np.cos(yy / 27.0)

    # 零件：只比背景亮 25 个灰阶
    body = (xx > 50) & (xx < 170) & (yy > 35) & (yy < 125)
    img[body] += 22

    # 划痕：更暗一点，宽 2 像素
    img[(np.abs(yy - 0.4 * xx - 60) < 1.5) & body] -= 15

    # 轻微噪声（真实相机一定有）
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# =====================================================================
# 二、直方图均衡化：用 CDF 当映射表
# =====================================================================

def equalize_hist(img):
    """
    【equalize_hist】histogram equalization → 直方图均衡化

    想法：让输出灰度的累积分布（CDF）变成一条直线。
    做法：把输入的 CDF 直接当映射表（LUT）：

        s = round( (L-1) · CDF(r) )
        CDF(r) = Σ_{k<=r} p(k)

    - L = 256（8 bit 灰度级数）
    - p(k)：灰度 k 出现的概率
    - s：映射后的灰度

    为什么单调：CDF 只增不减，所以不会把亮的变暗、暗的变亮。
    为什么能拉开：原来 p(k) 大的地方 CDF 上升快，映射后间隔大；
                 原来几乎没像素的灰度区间被"跳过"。
    代价：两个相邻灰阶可能被映射到同一级（量化），也可能把噪声放大。
    """
    hist = histogram(img)
    cdf = np.cumsum(hist) / img.size
    lut = np.round(255.0 * cdf).astype(np.uint8)
    return lut[img], lut, cdf


def demo_equalize():
    """
    【demo_equalize】contrast stretch → 对比度拉伸

    输入灰度集中在 [107, 143]，标准差很小；
    均衡化后铺满 [0, 255]，标准差显著变大。
    但注意：均衡化是**全局**的，它不知道"哪里是零件、哪里是背景"。
    """
    img = make_low_contrast()
    eq, lut, cdf = equalize_hist(img)

    print("输入：范围 [{}, {}]，均值 {:.1f}，标准差 {:.2f}，熵 {:.2f} bit，不同灰度 {} 级".format(
        img.min(), img.max(), img.mean(), img.std(), entropy(img), len(np.unique(img))))
    print("输出：范围 [{}, {}]，均值 {:.1f}，标准差 {:.2f}，熵 {:.2f} bit，不同灰度 {} 级".format(
        eq.min(), eq.max(), eq.mean(), eq.std(), entropy(eq), len(np.unique(eq))))

    print("映射表抽样：")
    for r in [0, 100, 110, 120, 130, 140, 255]:
        print("  输入 {:3d} -> 输出 {:3d}".format(r, lut[r]))

    # 均衡化的标志：输出的 CDF 尽量接近一条直线
    out_cdf = np.cumsum(histogram(eq)) / eq.size
    dev = np.abs(out_cdf - np.linspace(0, 1, 256)).max()
    print("输出 CDF 与直线的最大偏差 = {:.3f}".format(dev))

    # 注意：均衡化是单调映射，会把多个输入灰度合并成一个输出灰度，
    # 所以"熵"不一定上升（这里反而略降）。它保证的是"动态范围铺开"。
    assert eq.std() > 2 * img.std()
    assert eq.max() - eq.min() > 200
    assert np.all(np.diff(lut) >= 0)       # 单调不减
    assert entropy(eq) <= entropy(img) + 0.1
    return img, eq, lut


# =====================================================================
# 三、CLAHE：分块均衡 + 限幅 + 双线性插值
# =====================================================================

def clahe_lite(img, tiles=(8, 8), clip=2.0):
    """
    【clahe_lite】CLAHE → 限制对比度的自适应直方图均衡

    CLAHE = Contrast Limited Adaptive Histogram Equalization。

    三个动作：
        1. 分块（tile）：每块单独做均衡 -> 局部对比度提升
        2. 限幅（clip）：直方图超过 clip×平均高度的部分削掉，
           削掉的总量再平均分回每个 bin -> 噪声不会被放大到极端
        3. 插值（bilinear）：块与块之间双线性混合 -> 不出现方块接缝

    参数：
        tiles：把图切成 ty×tx 块
        clip ：限幅倍数（1 表示不限，常用 2~4）
    """
    img = np.asarray(img)
    h, w = img.shape
    ty, tx = tiles
    th, tw = int(np.ceil(h / ty)), int(np.ceil(w / tx))

    # 先补齐到 tile 的整数倍，边缘复制
    H, W = ty * th, tx * tw
    padded = np.pad(img, ((0, H - h), (0, W - w)), mode="edge")
    blocks = padded.reshape(ty, th, tx, tw).transpose(0, 2, 1, 3).reshape(ty, tx, th * tw)

    luts = np.zeros((ty, tx, 256), np.uint8)
    for i in range(ty):
        for j in range(tx):
            hist = histogram(blocks[i, j])
            limit = max(1, int(clip * th * tw / 256.0))
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist) / hist.sum()
            luts[i, j] = np.round(255 * cdf).astype(np.uint8)

    # 双线性权重：像素落在 4 个 tile 中心之间
    gy = np.arange(h) / th - 0.5
    gx = np.arange(w) / tw - 0.5
    y0 = np.clip(np.floor(gy).astype(int), 0, ty - 1)[:, None]
    x0 = np.clip(np.floor(gx).astype(int), 0, tx - 1)[None, :]
    y1 = np.clip(y0 + 1, 0, ty - 1)
    x1 = np.clip(x0 + 1, 0, tx - 1)
    wy = np.clip(gy - np.floor(gy), 0, 1)[:, None]
    wx = np.clip(gx - np.floor(gx), 0, 1)[None, :]

    v = img.astype(np.int64)
    out = ((1 - wy) * (1 - wx) * luts[y0, x0, v] +
           (1 - wy) * wx * luts[y0, x1, v] +
           wy * (1 - wx) * luts[y1, x0, v] +
           wy * wx * luts[y1, x1, v])
    return np.clip(out, 0, 255).astype(np.uint8)


def local_contrast(img, k=9):
    """
    【local_contrast】local contrast → 局部对比度

    用 k×k 窗口内的标准差衡量"这一小块有多少细节"。
    全图均衡化对平坦区没有帮助；CLAHE 会显著抬高局部标准差，
    代价是把噪声也一起抬起来。
    """
    from numpy.lib.stride_tricks import sliding_window_view
    r = k // 2
    padded = np.pad(np.asarray(img, np.float64), r, mode="reflect")
    win = sliding_window_view(padded, (k, k))
    return win.std(axis=(2, 3))


def demo_clahe(img_eq, img_raw):
    cla = clahe_lite(img_raw)
    lc_raw = local_contrast(img_raw)
    lc_eq = local_contrast(img_eq)
    lc_cla = local_contrast(cla)

    # 分开看"平坦背景"和"有结构的零件区"，否则标准差会被边缘主导
    smooth = img_raw < 120
    detail = img_raw > 125

    print("局部对比度（9×9 窗口标准差的均值）")
    print("           全图   平坦背景   零件区")
    print("  原图    {:6.2f}   {:6.2f}   {:6.2f}".format(
        lc_raw.mean(), lc_raw[smooth].mean(), lc_raw[detail].mean()))
    print("  全局均衡{:6.2f}   {:6.2f}   {:6.2f}".format(
        lc_eq.mean(), lc_eq[smooth].mean(), lc_eq[detail].mean()))
    print("  CLAHE   {:6.2f}   {:6.2f}   {:6.2f}".format(
        lc_cla.mean(), lc_cla[smooth].mean(), lc_cla[detail].mean()))
    print("结论：全局均衡把背景噪声放大了 {:.1f} 倍；CLAHE 只放大 {:.1f} 倍，".format(
        lc_eq[smooth].mean() / lc_raw[smooth].mean(),
        lc_cla[smooth].mean() / lc_raw[smooth].mean()))
    print("      但零件区仍然保留细节。")

    assert lc_cla[smooth].mean() < lc_eq[smooth].mean()
    assert lc_cla[detail].mean() > lc_raw[detail].mean()
    return cla


# =====================================================================
# 四、伽马：非线性点运算
# =====================================================================

def gamma_correct(img, gamma):
    """
    【gamma_correct】gamma correction → 伽马校正

        out = 255 · (in / 255)^γ

    - γ < 1：暗部被提亮（相机拍的东西"太黑"时用）
    - γ > 1：暗部被压暗、亮部更亮（"太灰"时用）
    - γ = 1：不变

    为什么不是线性：人眼对暗部的分辨力更强（近似对数），
    显示器的电光响应又近似 x^2.2；两边一凑，正好需要 γ 来补偿。
    """
    x = np.asarray(img, np.float64) / 255.0
    return np.clip(255.0 * np.power(x, gamma), 0, 255).astype(np.uint8)


def demo_gamma(img):
    print("gamma | 均值 | 标准差")
    print("------|------|-------")
    for g in [0.45, 1.0, 2.2]:
        out = gamma_correct(img, g)
        print(" {:4.2f} | {:5.1f} | {:6.2f}".format(g, out.mean(), out.std()))

    dark = gamma_correct(img, 0.45)
    bright = gamma_correct(img, 2.2)
    assert dark.mean() > img.mean() > bright.mean()
    return dark, bright


# =====================================================================
# 五、Otsu：让"类间方差"最大的阈值
# =====================================================================

def otsu_threshold(img):
    """
    【otsu_threshold】Otsu's method → 大津法

    把直方图按阈值 t 切成两类（背景 ≤ t、前景 > t），
    选让"类间方差"最大的 t：

        σ_B²(t) = ω0·ω1·(μ0 − μ1)²

    - ω0, ω1：两类的像素占比（和为 1）
    - μ0, μ1：两类的平均灰度
    - 类间方差越大，两类分得越开

    等价说法：Otsu 在找"直方图的双峰谷底"。
    它假设：图像基本是两类 + 噪声接近高斯 + 光照均匀。
    光照不均时直方图被抹平，Otsu 会给出一个"看起来合理但错误"的阈值。
    """
    hist = histogram(img)
    p = hist / hist.sum()
    levels = np.arange(256)

    omega = np.cumsum(p)                       # ω0(t)
    mu = np.cumsum(p * levels)                 # 累积一阶矩
    mu_t = mu[-1]

    # 防止除零：ω=0 或 ω=1 时类间方差记为 0
    denom = omega * (1 - omega)
    sigma_b = np.zeros(256)
    valid = denom > 0
    sigma_b[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]

    t = int(np.argmax(sigma_b))
    eta = sigma_b[t] / np.var(img) if np.var(img) > 0 else 0.0
    return t, sigma_b, float(eta)


def make_bimodal(h=120, w=160):
    """【make_bimodal】bimodal image → 双峰图（背景 80，零件 180，各带噪声）"""
    yy, xx = np.mgrid[0:h, 0:w]
    obj = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(h, w) * 0.28) ** 2
    img = np.where(obj, 180.0, 80.0)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), obj


def demo_otsu():
    img, obj = make_bimodal()
    t, sigma_b, eta = otsu_threshold(img)
    pred = img > t

    acc = (pred == obj).mean()
    print("真实阈值在 130 附近（80 与 180 的中点）")
    print("Otsu 阈值 = {}，类间方差 = {:.1f}，可分性 η = {:.3f}".format(t, sigma_b[t], eta))
    print("二值化准确率 = {:.2%}".format(acc))

    # 光照不均：左边暗、右边亮（乘性渐晕），直方图被抹成一片
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    shade = 0.35 + 1.00 * xx / img.shape[1]
    uneven = np.clip(img.astype(float) * shade, 0, 255).astype(np.uint8)
    t2, _, eta2 = otsu_threshold(uneven)
    acc2 = ((uneven > t2) == obj).mean()
    print("光照不均后：Otsu 阈值 = {}，可分性 η = {:.3f}，准确率 = {:.2%}".format(t2, eta2, acc2))
    print("（左侧暗、右侧亮，同一个零件在直方图上被摊成两个峰，单阈值救不了）")

    assert abs(t - 130) <= 12
    assert acc > 0.95
    assert eta2 < eta                     # 光照把可分性拉低
    assert acc2 < 0.95                    # 全局阈值在光照不均时失效
    return img, t, uneven, t2


# =====================================================================
# 六、画图
# =====================================================================

def plot_all(raw, eq, cla, gamma_dark, gamma_bright, bimodal, t, uneven, t2):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))

    ax = axes[0, 0]
    ax.imshow(raw, cmap="gray", vmin=0, vmax=255)
    ax.set_title("① 低对比度原图（灰度挤在 107~143）")

    ax = axes[0, 1]
    ax.hist(raw.ravel(), bins=64, range=(0, 255), color="#888")
    ax.set_title("② 直方图：一个窄峰，两端是空的")
    ax.set_xlabel("灰度")
    ax.set_ylabel("像素数")

    ax = axes[0, 2]
    ax.imshow(eq, cmap="gray", vmin=0, vmax=255)
    ax.set_title("③ 全局均衡化：对比度上去了，噪声也上去了")

    ax = axes[1, 0]
    ax.imshow(cla, cmap="gray", vmin=0, vmax=255)
    ax.set_title("④ CLAHE：局部对比度提升，不炸噪声")

    ax = axes[1, 1]
    x = np.linspace(0, 255, 256)
    for g, c in [(0.45, "#c33"), (1.0, "#333"), (2.2, "#36c")]:
        ax.plot(x, 255 * (x / 255) ** g, color=c, label="γ={}".format(g))
    ax.plot([0, 255], [0, 255], "k--", lw=0.6)
    ax.set_title("⑤ 伽马曲线：γ<1 提亮暗部")
    ax.set_xlabel("输入灰度")
    ax.set_ylabel("输出灰度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.hist(bimodal.ravel(), bins=64, range=(0, 255), color="#888")
    ax.axvline(t, color="#c33", label="Otsu t={}".format(t))
    ax.set_title("⑥ 双峰图：Otsu 找谷底")
    ax.set_xlabel("灰度")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "histogram.png")
        fig.savefig(path, dpi=120)
        print("\n图已保存:", path)
    else:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 60)
    print("一、低对比度图与直方图")
    raw = make_low_contrast()
    hist = histogram(raw)
    nz = np.nonzero(hist)[0]
    print("灰度非零区间: [{}, {}]，共 {} 级".format(nz[0], nz[-1], len(nz)))
    print("最大 bin 占比: {:.1%}".format(hist.max() / raw.size))

    print("\n" + "=" * 60)
    print("二、直方图均衡化")
    img, eq, lut = demo_equalize()

    print("\n" + "=" * 60)
    print("三、CLAHE")
    cla = demo_clahe(eq, img)

    print("\n" + "=" * 60)
    print("四、伽马")
    gamma_dark, gamma_bright = demo_gamma(img)

    print("\n" + "=" * 60)
    print("五、Otsu")
    bimodal, t, uneven, t2 = demo_otsu()

    print("\n" + "=" * 60)
    print("六、画图")
    plot_all(raw, eq, cla, gamma_dark, gamma_bright, bimodal, t, uneven, t2)

    print("\nALL PASS")


if __name__ == "__main__":
    main()