# -*- coding: utf-8 -*-
"""
S0 第 4 篇 —— 边缘检测与形态学：把"粗边"变成"能用的轮廓"

对应笔记：notes/S0-图像基础与工具链/4-边缘检测与形态学.md

这个脚本回答五个问题
--------------------
1. Sobel 出来的边缘为什么是"粗"的？非极大值抑制（NMS）在做什么？
2. 双阈值 + 滞后，为什么能把断掉的边缘接上、又去掉噪声点？
3. Canny 的五个步骤，每一步各解决什么问题？
4. 腐蚀 / 膨胀 / 开 / 闭，分别在什么场景用？
5. 连通域分析怎么把"一堆白点"变成"一个零件 + N 个噪点"？

运行：
    python 04_edges_morphology.py           # 弹窗
    python 04_edges_morphology.py --save    # 存到 out/edges_morphology.png
预期：最后打印 ALL PASS。
"""

import os
import sys

import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion
from numpy.lib.stride_tricks import sliding_window_view

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
# 一、基础算子（与第 2 篇同一套，不依赖外部库）
# =====================================================================

def correlate2d(img, kernel, padding=0):
    """【correlate2d】cross-correlation → 互相关（核不翻转）"""
    img = np.asarray(img, np.float64)
    kernel = np.asarray(kernel, np.float64)
    if padding > 0:
        img = np.pad(img, padding, mode="constant")
    kh, kw = kernel.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    win = sliding_window_view(img, (kh, kw))
    return np.einsum("ijkl,kl->ij", win, kernel)


def gaussian_kernel(sigma, radius=None):
    """【gaussian_kernel】2-D Gaussian → 二维高斯核（外积，和为 1）"""
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def blur(img, sigma=1.4):
    """【blur】Gaussian blur → 高斯模糊（补零边界，够用即可）"""
    g = gaussian_kernel(sigma)
    r = g.shape[0] // 2
    return correlate2d(img, g, padding=r)


def sobel(img):
    """
    【sobel】Sobel gradient → 索贝尔梯度

        Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
        Gy = Gx.T
        |G| = hypot(Gx, Gy)
        θ   = atan2(Gy, Gx)   单位：弧度，范围 (-π, π]

    θ 指向"灰度增长最快"的方向，也就是**垂直穿过边缘**的方向。
    Canny 的 NMS 就要沿这个方向找最大值。
    """
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    gx = correlate2d(img, kx, padding=1)
    gy = correlate2d(img, kx.T, padding=1)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


# =====================================================================
# 二、非极大值抑制：把粗边压成单像素
# =====================================================================

def non_max_suppression(mag, ang):
    """
    【non_max_suppression】NMS → 非极大值抑制

    问题：Sobel 的边缘有 3~5 像素宽，因为一个阶跃会在好几个位置上都有响应。

    做法：对每个像素，沿**梯度方向**比较前后两个邻居：
        只有自己是局部最大才保留，否则置 0。

    方向先量化成 4 个箱子（0°/45°/90°/135°），
    因为像素只有 8 个邻居，取最近的 2 个：

        0°   -> 左右
        45°  -> 左上 / 右下
        90°  -> 上下
        135° -> 右上 / 左下

    注意：方向是"梯度方向"（垂直边缘），不是边缘走向。
    """
    padded = np.pad(np.asarray(mag, np.float64), 1, mode="constant")
    c = padded[1:-1, 1:-1]
    east, west = padded[1:-1, 2:], padded[1:-1, :-2]
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    ne, sw = padded[:-2, 2:], padded[2:, :-2]
    nw, se = padded[:-2, :-2], padded[2:, 2:]

    # 把角度折到 [0,180)，再除以 45 取整
    q = np.round((np.degrees(ang) % 180) / 45.0).astype(int) % 4

    n1 = np.select([q == 0, q == 1, q == 2, q == 3], [east, se, south, sw])
    n2 = np.select([q == 0, q == 1, q == 2, q == 3], [west, nw, north, ne])

    keep = (c >= n1) & (c >= n2)
    return np.where(keep, c, 0.0)


# =====================================================================
# 三、双阈值 + 滞后：噪声点和断边一起处理
# =====================================================================

def hysteresis(mag, low, high):
    """
    【hysteresis】hysteresis thresholding → 滞后阈值

    两个阈值：
        强边缘：|G| ≥ high   -> 一定是边缘
        弱边缘：low ≤ |G| < high -> 可能是边缘

    规则：只有"和强边缘连通"的弱边缘才保留。
    实现：对弱边缘做 8 连通标记，只保留包含强边缘的连通块。

    为什么有效：
        - 噪声点是孤立的 -> 不连强边缘 -> 丢掉
        - 真边缘虽然中间衰减到弱，但两端连着强 -> 整条留下
    这就是"滞后（hysteresis）"：阈值历史决定当前是开还是关。
    """
    strong = mag >= high
    weak = mag >= low
    lab, n = label(weak, structure=np.ones((3, 3)))
    keep = np.unique(lab[strong])
    keep = keep[keep > 0]
    return np.isin(lab, keep), strong, n


# =====================================================================
# 四、Canny 五步
# =====================================================================

def canny_lite(img, sigma=1.4, low=30, high=150):
    """
    【canny_lite】Canny pipeline → Canny 五步

        1. 高斯模糊        降噪（否则梯度全是噪点）
        2. Sobel 梯度      幅值 + 方向
        3. 非极大值抑制    粗边 -> 单像素
        4. 双阈值          分强/弱
        5. 滞后            弱边连强边才保留

    返回每一步的中间结果，方便对照"哪一步去掉了什么"。
    """
    b = blur(img, sigma)
    gx, gy, mag, ang = sobel(b)
    thin = non_max_suppression(mag, ang)
    edges, strong, n = hysteresis(thin, low, high)
    return {
        "blur": b,
        "mag": mag,
        "thin": thin,
        "edges": edges,
        "strong": strong,
        "n_weak_components": n,
    }


def demo_canny(img):
    res = canny_lite(img)
    mag, thin, edges = res["mag"], res["thin"], res["edges"]

    # 用同一阈值统计"抑制前 / 抑制后 / 滞后后"的像素数
    raw_px = int((mag > 30).sum())
    thin_px = int((thin > 30).sum())
    strong_px = int(res["strong"].sum())
    final_px = int(edges.sum())

    print("阈值 low=30 / high=150（梯度幅值）")
    print("Sobel 幅值 >30 的像素 :", raw_px)
    print("NMS 后 >30 的像素     :", thin_px, "（压缩到 {:.1f}%）".format(100 * thin_px / raw_px))
    print("强边缘像素            :", strong_px)
    print("弱边缘连通块数        :", res["n_weak_components"])
    print("滞后后最终边缘像素    :", final_px, "（是强边缘的 {:.2f} 倍）".format(final_px / strong_px))
    print("边缘占全图比例        : {:.2%}".format(final_px / img.size))

    assert thin_px < raw_px * 0.6          # NMS 至少压掉 40%
    assert final_px > strong_px            # 滞后补回了弱边
    assert final_px < thin_px
    return res


# =====================================================================
# 五、形态学：腐蚀、膨胀、开、闭
# =====================================================================

def erode(binary, k=3):
    """
    【erode】erosion → 腐蚀

    窗口内**全为 1** 才保留：把前景"削掉一圈"。
    作用：去小噪点、断开细连接、让目标变小。
    """
    r = k // 2
    p = np.pad(np.asarray(binary, bool), r, mode="constant", constant_values=False)
    win = sliding_window_view(p, (k, k))
    return win.all(axis=(2, 3))


def dilate(binary, k=3):
    """
    【dilate】dilation → 膨胀

    窗口内**有 1** 就置 1：把前景"长一圈"。
    作用：填补小孔、连接断裂、让目标变大。
    """
    r = k // 2
    p = np.pad(np.asarray(binary, bool), r, mode="constant", constant_values=False)
    win = sliding_window_view(p, (k, k))
    return win.any(axis=(2, 3))


def opening(binary, k=3):
    """【opening】opening → 开运算：先腐蚀后膨胀，去小噪点、保大形状"""
    return dilate(erode(binary, k), k)


def closing(binary, k=3):
    """【closing】closing → 闭运算：先膨胀后腐蚀，填小孔、连断缝"""
    return erode(dilate(binary, k), k)


def demo_morphology():
    """
    【demo_morphology】morphology → 形态学

    造一个"有噪点、有孔洞"的二值图：
        - 主体：矩形 + 圆（真目标）
        - 椒盐噪点：随机白点（假目标）
        - 孔洞：主体内部的黑点

    开运算去掉孤立白点；闭运算填掉内部孔洞。
    """
    # 与 scipy 对照，确认手写没错
    a = rng.random((40, 50)) > 0.5
    assert np.array_equal(erode(a, 3), binary_erosion(a, structure=np.ones((3, 3))))
    assert np.array_equal(dilate(a, 3), binary_dilation(a, structure=np.ones((3, 3))))
    print("手写腐蚀/膨胀与 scipy 完全一致")

    h, w = 160, 220
    binary = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    binary |= (xx > 40) & (xx < 150) & (yy > 35) & (yy < 125)
    binary |= (xx - 170) ** 2 + (yy - 100) ** 2 < 28 ** 2

    # 内部小孔（黑）：半径 3，闭运算能补上
    binary[(xx - 90) ** 2 + (yy - 80) ** 2 < 3 ** 2] = False
    # 椒盐噪声（白）
    noise = rng.random((h, w)) < 0.004
    noisy = binary | noise

    # 再挖掉一些边界，模拟阈值造成的缺口
    noisy[(yy > 120) & (yy < 126) & (xx > 80) & (xx < 110)] = False

    op = opening(noisy, 3)
    cl = closing(op, 7)

    print("噪声图白像素   :", int(noisy.sum()))
    print("开运算后白像素 :", int(op.sum()), "（去掉孤立噪点）")
    print("闭运算后白像素 :", int(cl.sum()), "（补回小孔，+{} 像素）".format(int(cl.sum() - op.sum())))
    print("真值白像素     :", int(binary.sum()))

    assert op.sum() < noisy.sum()
    assert cl.sum() > op.sum()
    return binary, noisy, op, cl


# =====================================================================
# 六、连通域：从"白点"到"对象"
# =====================================================================

def demo_components(noisy, cleaned, true_binary):
    """
    【demo_components】connected components → 连通域

    label() 把相邻的 1 编成同一个号（8 连通 = 斜着也算相连）。
    然后按面积过滤：
        面积 < min_area  -> 当噪声丢掉
        面积 >= min_area -> 当一个对象

    输出的 stats 里还有 bbox（外接框）和质心，后面做检测/测量都要用。
    """
    from scipy.ndimage import find_objects, center_of_mass

    for name, mask in [("噪声图", noisy), ("开运算后", cleaned)]:
        lab, n = label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(lab.ravel())[1:]
        keep = areas >= 20
        print("{}：{} 个连通块，面积≥20 的有 {} 个，最大面积 {}".format(
            name, n, int(keep.sum()), int(areas.max())))

    lab, n = label(cleaned, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel())[1:]
    order = np.argsort(areas)[::-1] + 1
    objs = find_objects(lab)
    centers = center_of_mass(cleaned, lab, order)

    print("按面积排序（前 3）：")
    for rank, idx in enumerate(order[:3]):
        sl = objs[idx - 1]
        cy, cx = centers[rank]
        print("  #{} 面积={:5d} bbox=(y{}:{}, x{}:{}) 质心=({:.1f}, {:.1f})".format(
            rank + 1, int(areas[idx - 1]),
            sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, cx, cy))

    assert areas.max() > 1000
    assert int((areas >= 20).sum()) <= 3
    return lab, areas


def filter_small(lab, min_area=20):
    """【filter_small】size filter → 按面积过滤：小连通块置 0"""
    areas = np.bincount(lab.ravel())
    keep = np.zeros(areas.size, bool)
    keep[1:] = areas[1:] >= min_area
    return keep[lab]


# =====================================================================
# 七、造图与画图
# =====================================================================

def make_scene(h=180, w=240):
    """【make_scene】synthetic scene → 合成场景（方块 + 圆 + 斜划痕 + 噪声）"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 55 + 35 * xx / w

    img[(xx > 40) & (xx < 130) & (yy > 40) & (yy < 140)] = 195
    img[(xx - 185) ** 2 + (yy - 95) ** 2 < 36 ** 2] = 160
    img[np.abs(yy - (0.35 * xx + 50)) < 1.8] = 240
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255)


def plot_all(img, res, true_binary, noisy, op, cl, lab):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("① 原图")

    ax = axes[0, 1]
    ax.imshow(res["mag"], cmap="gray")
    ax.set_title("② Sobel 幅值：边缘是粗的")

    ax = axes[0, 2]
    ax.imshow(res["thin"], cmap="gray")
    ax.set_title("③ NMS 后：压成单像素")

    ax = axes[0, 3]
    ax.imshow(res["edges"], cmap="gray")
    ax.set_title("④ 双阈值 + 滞后：最终边缘")

    ax = axes[1, 0]
    ax.imshow(noisy, cmap="gray")
    ax.set_title("⑤ 含噪二值图（白点 = 噪声）")

    ax = axes[1, 1]
    ax.imshow(op, cmap="gray")
    ax.set_title("⑥ 开运算：去掉孤立白点")

    ax = axes[1, 2]
    ax.imshow(cl, cmap="gray")
    ax.set_title("⑦ 闭运算：补回孔洞")

    ax = axes[1, 3]
    rng_palette = rng.integers(40, 255, size=(lab.max() + 1, 3)).astype(np.uint8)
    rng_palette[0] = 0
    ax.imshow(rng_palette[lab])
    ax.set_title("⑧ 连通域着色：{} 个对象".format(int(lab.max())))

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    if SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "edges_morphology.png")
        fig.savefig(path, dpi=120)
        print("\n图已保存:", path)
    else:
        plt.show()
    plt.close(fig)


def main():
    img = make_scene()

    print("=" * 60)
    print("一、Canny 五步")
    res = demo_canny(img)

    print("\n" + "=" * 60)
    print("二、形态学")
    true_binary, noisy, op, cl = demo_morphology()

    print("\n" + "=" * 60)
    print("三、连通域")
    lab, areas = demo_components(noisy, cl, true_binary)

    print("\n" + "=" * 60)
    print("四、画图")
    plot_all(img, res, true_binary, noisy, op, cl, lab)

    print("\nALL PASS")


if __name__ == "__main__":
    main()
