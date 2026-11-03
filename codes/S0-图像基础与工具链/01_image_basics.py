# -*- coding: utf-8 -*-
"""
S0 第 1 篇 —— 图像是什么：形状、通道、灰度、量化、采样

对应笔记：notes/S0-图像基础与工具链/1-图像是什么.md

这个脚本回答五个问题
--------------------
1. 一张"彩色图"在内存里到底是什么？（shape / dtype / 字节数）
2. BGR 和 RGB 只差一个顺序，为什么会画出"蓝脸"？
3. 灰度 = 三通道平均吗？为什么不是？
4. 把 256 级压到 2 级，损失有多大？（PSNR 说话）
5. 缩小图像为什么必须先滤波？直接隔点取样会怎样？

运行：
    python 01_image_basics.py           # 弹窗看 6 张图
    python 01_image_basics.py --save    # 存到 out/image_basics.png
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
from matplotlib.colors import rgb_to_hsv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# =====================================================================
# 一、造一张"零件"图：有颜色、有渐变、有细节
# =====================================================================

def make_part_image(h=180, w=240):
    """
    【make_part_image】synthetic image → 合成图

    造一张 RGB 浮点图（范围 [0,1]），包含：
        - 背景：从左到右的灰阶渐变（模拟光照不均）
        - 零件本体：橙色矩形
        - 圆孔：绿色
        - 矩形内的细条纹：蓝色
        - 左上角高光：接近白色

    为什么不用真实照片：固定种子 + 无外部文件，任何机器结果一致。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 背景渐变 40 -> 90（除以 255 变成 [0,1]）
    img = np.zeros((h, w, 3), dtype=np.float32)
    base = (40 + 50 * xx / w) / 255.0
    img[..., 0] = base
    img[..., 1] = base
    img[..., 2] = base

    # 零件本体：橙色 (R=200, G=90, B=60)
    body = (xx > 55) & (xx < 165) & (yy > 45) & (yy < 145)
    img[body] = np.array([200, 90, 60], np.float32) / 255.0

    # 圆孔：绿色 (R=40, G=180, B=90)
    hole = (xx - 190) ** 2 + (yy - 100) ** 2 < 24 ** 2
    img[hole] = np.array([40, 180, 90], np.float32) / 255.0

    # 细条纹：蓝色 (R=60, G=80, B=200)，只在零件内部
    stripe = body & (((xx.astype(int) // 6) % 2) == 0) & (yy > 65) & (yy < 125)
    img[stripe] = np.array([60, 80, 200], np.float32) / 255.0

    # 高光：一块接近饱和的白
    high = (xx - 30) ** 2 + (yy - 30) ** 2 < 10 ** 2
    img[high] = np.array([250, 248, 245], np.float32) / 255.0

    return np.clip(img, 0, 1)


def to_u8(img_f):
    """【to_u8】float [0,1] -> uint8 [0,255]，四舍五入后截断"""
    return np.clip(np.round(img_f * 255.0), 0, 255).astype(np.uint8)


# =====================================================================
# 二、形状、dtype、内存
# =====================================================================

def demo_storage(img_f):
    """
    【demo_storage】storage layout → 内存布局

    彩色图 = 高 × 宽 × 通道，uint8 每像素 3 字节。
    180×240×3 = 129,600 字节 ≈ 127 KB。

    注意：内存里没有"行/列"标签，只有步长（stride）：
        往下一行 = 跳过 W×3 个元素
        往右一列 = 跳过 3 个元素
    这也是"为什么图像是行优先（row-major）"。
    """
    u8 = to_u8(img_f)
    h, w, c = u8.shape
    print("shape          :", u8.shape)
    print("dtype          :", u8.dtype, " itemsize:", u8.itemsize, "字节")
    print("像素总数       :", h * w, " 通道数:", c)
    print("总字节         :", u8.nbytes, "=", round(u8.nbytes / 1024, 1), "KB")
    print("strides (字节) :", u8.strides, " <- (行步长, 列步长, 通道步长)")
    print("值域           : [{}, {}]".format(u8.min(), u8.max()))

    assert u8.shape == (180, 240, 3)
    assert u8.nbytes == 180 * 240 * 3
    assert u8.strides == (240 * 3, 3, 1)
    return u8


# =====================================================================
# 三、BGR vs RGB：只差顺序，后果很大
# =====================================================================

def demo_bgr_rgb(img_u8):
    """
    【demo_bgr_rgb】channel order → 通道顺序

    OpenCV 默认按 BGR 存：img[..., 0] 是蓝，不是红。
    matplotlib / PIL 按 RGB。

    只差一个顺序，但：
        - 通道均值会整体错位（红蓝互换）
        - 直接 imshow 会得到"蓝脸"
        - 灰度权重会乘错通道，亮度全错

    修复只有一行：rgb = bgr[..., ::-1]（或 cv2.cvtColor）。
    """
    bgr = img_u8[..., ::-1]          # 模拟 OpenCV 读进来的 BGR
    rgb_means = img_u8.reshape(-1, 3).mean(axis=0)
    bgr_means = bgr.reshape(-1, 3).mean(axis=0)

    print("RGB 通道均值 : R={:.2f} G={:.2f} B={:.2f}".format(*rgb_means))
    print("BGR 通道均值 : B={:.2f} G={:.2f} R={:.2f}".format(*bgr_means))

    # 取一个已知像素：零件本体 (200, 90, 60)
    r, g, b = img_u8[90, 100]
    print("像素 (x=100, y=90) 的 RGB = ({}, {}, {})".format(r, g, b))

    # 如果误把 BGR 当 RGB 用，读到的"红"其实是蓝
    br, bg, bb = bgr[90, 100]
    print("同一位置按 BGR 读     = ({}, {}, {}) <- 顺序反了".format(br, bg, bb))

    assert rgb_means[0] > rgb_means[2]        # 这张图整体偏红
    assert np.allclose(bgr_means, rgb_means[::-1])
    assert bgr[90, 100, 0] == img_u8[90, 100, 2]   # BGR 的第 0 通道 = RGB 的 B


# =====================================================================
# 四、灰度：加权，不是平均
# =====================================================================

def demo_grayscale(img_u8):
    """
    【demo_grayscale】luminance → 亮度（灰度）

    人眼对绿最敏感、蓝最不敏感。ITU-R BT.601 的权重：
        Y = 0.299 R + 0.587 G + 0.114 B

    对比"三通道直接平均"：
        纯绿 (0,255,0)：加权 149.7，平均 85.0 —— 差 64.7
        纯蓝 (0,0,255)：加权 29.1，平均 85.0 —— 差 55.9

    结论：平均灰度会把"亮绿"和"亮蓝"判成一样亮，
    而实际视觉上绿色亮得多。这就是为什么灰度化不能用 mean。
    """
    rgb = img_u8.astype(np.float32)
    weighted = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    average = rgb.mean(axis=2)

    diff = np.abs(weighted - average)
    print("加权 vs 平均：平均差 {:.2f} / 255，最大差 {:.1f}".format(diff.mean(), diff.max()))

    # 用三个纯色验证权重方向
    for name, c in [("红", (255, 0, 0)), ("绿", (0, 255, 0)), ("蓝", (0, 0, 255))]:
        c = np.array(c, np.float32)
        y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        print("纯{}: 加权灰度 {:6.1f}，平均灰度 {:6.1f}".format(name, y, c.mean()))

    assert np.isclose(0.299 * 255 + 0.587 * 255 + 0.114 * 255, 255.0)
    assert weighted.max() <= 255.0001
    return weighted / 255.0, average / 255.0


# =====================================================================
# 五、HSV：把"颜色"和"亮度"拆开
# =====================================================================

def demo_hsv(img_f):
    """
    【demo_hsv】HSV → 色相 / 饱和度 / 明度

    H（Hue，色相）：颜色在色环上的角度，0~1 对应 0°~360°
    S（Saturation，饱和度）：颜色有多纯，0=灰，1=纯色
    V（Value，明度）：最亮通道的值，0=黑

    为什么分割常用 HSV：光照变化主要改 V，不太改 H；
    在 RGB 里，一开灯三个通道一起变，阈值全废。
    """
    hsv = rgb_to_hsv(img_f)
    print("H 范围 : [{:.3f}, {:.3f}]（0~1 对应 0°~360°）".format(hsv[..., 0].min(), hsv[..., 0].max()))
    print("S 范围 : [{:.3f}, {:.3f}]".format(hsv[..., 1].min(), hsv[..., 1].max()))
    print("V 范围 : [{:.3f}, {:.3f}]".format(hsv[..., 2].min(), hsv[..., 2].max()))

    # 零件本体的颜色：橙色 -> 色相约 0.05
    h, s, v = hsv[90, 100]
    print("零件像素 HSV = ({:.3f}, {:.3f}, {:.3f}) -> 色相 {:.0f}°".format(h, s, v, h * 360))

    # 纯色卡：红绿蓝的色相分别落在 0 / 1/3 / 2/3
    pure = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], np.float32)
    hues = rgb_to_hsv(pure)[0, :, 0]
    print("纯红/绿/蓝色相 :", np.round(hues, 3), "（0, 0.333, 0.667）")

    assert np.allclose(hues, [0, 1 / 3, 2 / 3], atol=1e-6)
    return hsv


# =====================================================================
# 六、量化：256 级 -> k 级，损失多少
# =====================================================================

def quantize(img_u8, levels):
    """
    【quantize】quantization → 量化

    把 [0,255] 均匀切成 levels 档：
        q = round(x / 255 * (levels-1)) / (levels-1) * 255

    levels=2 只剩 0/255，就是"二值图"（但这是均匀切，不是阈值分割）。
    量化误差的上界是半个台阶：255/(2*(levels-1))。
    """
    step = 255.0 / (levels - 1)
    q = np.round(img_u8 / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


def psnr(a, b):
    """
    【psnr】Peak Signal-to-Noise Ratio → 峰值信噪比

        MSE = mean((a-b)^2)
        PSNR = 20 log10(255 / sqrt(MSE))  单位 dB

    只对"同尺寸、同内容"的失真有效；越大越好，>30 dB 通常肉眼看不出。
    """
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def demo_quantization(img_u8):
    """
    【demo_quantization】bit depth → 位深

    8 bit = 256 级是工业相机常见输出；
    压到 4 bit（16 级）还能看，2 bit（4 级）就出现明显"色带"。
    这就是为什么"省带宽"不能随便降位深。
    """
    print("levels | 台阶 | PSNR(dB) | 最大误差")
    print("-------|------|----------|---------")
    result = {}
    for levels in [2, 4, 8, 16, 32]:
        q = quantize(img_u8, levels)
        p = psnr(img_u8, q)
        err = int(np.abs(q.astype(int) - img_u8.astype(int)).max())
        result[levels] = (q, p)
        print("  {:3d}  | {:4.1f} |  {:6.2f}  |   {:3d}".format(
            levels, 255.0 / (levels - 1), p, err))

    assert result[2][1] < result[16][1]      # 级数越多，PSNR 越高
    assert result[2][0].max() == 255 and result[2][0].min() == 0
    return result


# =====================================================================
# 七、采样：缩小时为什么必须先滤波
# =====================================================================

def make_fine_stripes(h=60, w=240, period=4):
    """
    【make_fine_stripes】aliasing pattern → 抗混叠测试图

    每 period 个像素一条白线（其余为黑）：
        真实均值 = 255 / period = 63.75
    若直接每隔 period 取一个点，取到的全是白线 -> 均值 255。
    这就是"摩尔纹 / 混叠"的最小例子。
    """
    xx = np.arange(w)
    img = np.zeros((h, w), np.float32)
    img[:, (xx % period) == 0] = 1.0
    return img


def demo_downsample():
    """
    【demo_downsample】downsampling / aliasing → 下采样与混叠

    缩小 k 倍有两条路：
        A. 直接隔 k 个取一个（点采样）：快，但高频会"折叠"成假低频
        B. 先低通（均值/高斯）再取样：慢一点，但不会凭空造出条纹

    奈奎斯特（Nyquist）要求：采样率 > 2×最高频率。
    细条纹周期 4，采样步长 4 -> 采样率刚好等于信号频率，必然混叠。
    """
    period = 4
    img = make_fine_stripes(period=period)
    true_mean = img.mean() * 255

    # A. 点采样：每 4 个取 1 个
    sampled = img[:, ::period]
    sampled_mean = sampled.mean() * 255

    # B. 先 4×4 均值（盒式低通）再取样
    h, w = img.shape
    block = img[:h // period * period, :w // period * period]
    block = block.reshape(h // period, period, w // period, period).mean(axis=(1, 3))
    block_mean = block.mean() * 255

    print("真实均值       : {:.2f}".format(true_mean))
    print("直接隔点取样   : {:.2f}  <- 偏亮 {:.1f} 倍".format(sampled_mean, sampled_mean / true_mean))
    print("先均值再取样   : {:.2f}  <- 和真值一致".format(block_mean))

    assert abs(sampled_mean - 255) < 1e-6
    assert abs(block_mean - true_mean) < 1e-6
    return img, sampled


# =====================================================================
# 八、画图：一张图只讲一件事
# =====================================================================

def plot_all(img_f, img_u8, gray_w, gray_avg, quant_res, stripes, sampled):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))

    ax = axes[0, 0]
    ax.imshow(img_f)
    ax.set_title("① 合成图：渐变背景 + 彩色零件")
    ax.set_xlabel("x（列，向右）")
    ax.set_ylabel("y（行，向下）")

    ax = axes[0, 1]
    means = img_u8.reshape(-1, 3).mean(axis=0)
    ax.bar(["R", "G", "B"], means, color=["#d33", "#3a3", "#33d"])
    ax.set_title("② 通道均值：BGR 只是顺序不同")
    ax.set_ylabel("平均像素值")
    for i, m in enumerate(means):
        ax.text(i, m + 1, "{:.1f}".format(m), ha="center", fontsize=9)

    ax = axes[0, 2]
    ax.imshow(np.abs(gray_w - gray_avg), cmap="inferno")
    ax.set_title("③ |加权灰度 − 平均灰度|：绿色差最多")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.imshow(quant_res[2][0])
    ax.set_title("④ 2 bit 量化（4 级）：出现色带，PSNR={:.1f} dB".format(quant_res[2][1]))

    ax = axes[1, 1]
    ax.imshow(stripes, cmap="gray", aspect="auto")
    ax.set_title("⑤ 周期 4 的细条纹（真实均值 63.75）")

    ax = axes[1, 2]
    ax.plot([0, 1, 2], [255, 63.75, 63.75], "o-", label="")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["隔点取样", "先滤波", "真值"])
    ax.set_ylim(0, 280)
    ax.set_title("⑥ 缩小 4 倍：隔点取样把亮度抬到 255")
    ax.set_ylabel("平均像素值")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "image_basics.png")
        fig.savefig(path, dpi=120)
        print("\n图已保存:", path)
    else:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 60)
    print("一、造图")
    img_f = make_part_image()
    print("造图完成:", img_f.shape, img_f.dtype)

    print("\n" + "=" * 60)
    print("二、形状 / dtype / 内存")
    img_u8 = demo_storage(img_f)

    print("\n" + "=" * 60)
    print("三、BGR vs RGB")
    demo_bgr_rgb(img_u8)

    print("\n" + "=" * 60)
    print("四、灰度：加权 vs 平均")
    gray_w, gray_avg = demo_grayscale(img_u8)

    print("\n" + "=" * 60)
    print("五、HSV")
    demo_hsv(img_f)

    print("\n" + "=" * 60)
    print("六、量化")
    quant_res = demo_quantization(img_u8)

    print("\n" + "=" * 60)
    print("七、下采样与混叠")
    stripes, sampled = demo_downsample()

    print("\n" + "=" * 60)
    print("八、画图")
    plot_all(img_f, img_u8, gray_w, gray_avg, quant_res, stripes, sampled)

    print("\nALL PASS")


if __name__ == "__main__":
    main()