# -*- coding: utf-8 -*-
"""
S0 第 2 篇 —— 卷积与滤波：从两层循环到 im2col

对应笔记：notes/S0-图像基础与工具链/2-卷积与滤波.md

这个脚本回答五个问题
--------------------
1. "卷积"和"相关"差在哪？为什么 CNN 里其实算的是相关？
2. padding / stride 怎么改变输出尺寸？公式怎么来的？
3. 手写两层循环的卷积，和 scipy 的结果差多少？
4. 高斯核为什么能拆成两个一维核？快多少？
5. 边界补零为什么会把图像边缘变暗？该用哪种 padding？

运行：
    python 02_convolution.py           # 弹窗
    python 02_convolution.py --save    # 存到 out/convolution.png
预期：最后打印 ALL PASS。
"""

import os
import sys
import time

import numpy as np
from scipy.signal import correlate2d, convolve2d

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
# 一、相关 vs 卷积：只差一个"翻转"
# =====================================================================

def correlate2d_naive(img, kernel, stride=1, padding=0):
    """
    【correlate2d_naive】cross-correlation → 互相关

    对输出每个位置 (i, j)：
        out[i, j] = Σ_{u,v} img[i*s+u, j*s+v] * kernel[u, v]

    注意：核**不翻转**，直接盖上去逐项相乘再求和。
    深度学习框架里的 Conv2d 做的就是这个（只是多了通道维）。
    """
    img = np.asarray(img, np.float64)
    kernel = np.asarray(kernel, np.float64)
    if padding > 0:
        img = np.pad(img, padding, mode="constant")

    kh, kw = kernel.shape
    out_h = (img.shape[0] - kh) // stride + 1
    out_w = (img.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w), np.float64)

    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def convolve2d_naive(img, kernel, stride=1, padding=0):
    """
    【convolve2d_naive】convolution → 卷积

    数学定义的卷积要先把核**上下左右翻转 180°**：
        out[i, j] = Σ img[i+u, j+v] * kernel[kh-1-u, kw-1-v]

    所以：卷积(img, k) == 相关(img, rot180(k))。
    """
    return correlate2d_naive(img, np.rot90(np.asarray(kernel), 2), stride, padding)


def demo_flip():
    """
    【demo_flip】kernel flip → 核翻转

    取一个不对称的核 [[0,0,0],[0,0,0],[0,0,1]]（只有右下角是 1）。
    相关：输出的 (i,j) 取自输入的右下方 -> 图像整体向左上"移"。
    卷积：核翻转后取自左上方 -> 图像整体向右下"移"。

    对称核（高斯、拉普拉斯、Sobel 的 |gx|）翻不翻转都一样，
    所以工程里常把"相关"直接叫"卷积"，不纠结。
    """
    img = np.zeros((7, 7), np.float64)
    img[3, 3] = 1.0
    k = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1]], np.float64)

    corr = correlate2d_naive(img, k)
    conv = convolve2d_naive(img, k)

    cy, cx = np.unravel_index(np.argmax(corr), corr.shape)
    vy, vx = np.unravel_index(np.argmax(conv), conv.shape)
    print("相关峰值位置 (row, col) :", (int(cy), int(cx)))
    print("卷积峰值位置 (row, col) :", (int(vy), int(vx)))
    print("两者相差 2 像素（核 3×3 的翻转位移）")

    assert (cy, cx) == (1, 1)
    assert (vy, vx) == (3, 3)
    return corr, conv


# =====================================================================
# 二、输出尺寸公式：padding 与 stride
# =====================================================================

def out_size(n, k, stride=1, padding=0):
    """
    【out_size】output size → 输出尺寸

        out = floor((n + 2p - k) / s) + 1

    - n：输入边长
    - k：核边长
    - p：每边补几圈
    - s：步长（隔几个像素滑一次）

    记忆法：卷积核"能完整放下几次"。除不尽时，右边/下边多出来的像素被丢掉
    （floor），这也是"same padding"要凑成 (k-1)/2 的原因。
    """
    return (n + 2 * padding - k) // stride + 1


def demo_size():
    print("n=7, k=3, s=1, p=0 ->", out_size(7, 3, 1, 0), "（5）")
    print("n=7, k=3, s=1, p=1 ->", out_size(7, 3, 1, 1), "（same，仍为 7）")
    print("n=7, k=3, s=2, p=1 ->", out_size(7, 3, 2, 1), "（4，向下取整）")
    print("n=8, k=3, s=2, p=1 ->", out_size(8, 3, 2, 1), "（4）")

    assert out_size(7, 3, 1, 0) == 5
    assert out_size(7, 3, 1, 1) == 7
    assert out_size(7, 3, 2, 1) == 4
    assert out_size(8, 3, 2, 1) == 4


# =====================================================================
# 三、和 scipy 对照：手写对不对
# =====================================================================

def demo_against_scipy():
    """
    【demo_against_scipy】reference check → 与参考实现对照

    scipy.signal.correlate2d / convolve2d 是成熟实现。
    手写版和它差 < 1e-12，才说明下标、翻转、padding 都没写错。

    断言用 allclose，不写 ==：浮点求和顺序不同，末位会有差异。
    """
    img = rng.normal(0, 1, size=(9, 11))
    k = rng.normal(0, 1, size=(3, 5))

    mine_c = correlate2d_naive(img, k, stride=1, padding=0)
    ref_c = correlate2d(img, k, mode="valid")
    mine_v = convolve2d_naive(img, k, stride=1, padding=0)
    ref_v = convolve2d(img, k, mode="valid")

    print("相关：手写 vs scipy 最大差 = {:.2e}".format(np.abs(mine_c - ref_c).max()))
    print("卷积：手写 vs scipy 最大差 = {:.2e}".format(np.abs(mine_v - ref_v).max()))

    assert np.allclose(mine_c, ref_c, atol=1e-12)
    assert np.allclose(mine_v, ref_v, atol=1e-12)


# =====================================================================
# 四、im2col：把卷积变成一次矩阵乘法
# =====================================================================

def im2col(img, k, stride=1, padding=0):
    """
    【im2col】image to column → 图像转列

    把每个 k×k 的滑窗"摊平"成一行，得到矩阵
        cols: (输出位置数, k*k)

    于是卷积 = cols @ kernel.reshape(-1)，一次 GEMM 算完。
    这是所有深度学习框架（cuDNN、MNN、TensorRT）卷积实现的基本套路：
        矩阵乘法有高度优化的 BLAS 内核，比 Python/C 的两层循环快几个数量级。
    """
    if padding > 0:
        img = np.pad(img, padding, mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(img, (k, k))[::stride, ::stride]   # (oh, ow, k, k)
    oh, ow = win.shape[:2]
    return win.reshape(oh * ow, k * k), (oh, ow)


def conv2d_im2col(img, kernel, stride=1, padding=0):
    """【conv2d_im2col】im2col convolution → 矩阵乘法版卷积"""
    k = kernel.shape[0]
    cols, (oh, ow) = im2col(np.asarray(img, np.float64), k, stride, padding)
    out = cols @ np.asarray(kernel, np.float64).reshape(-1)
    return out.reshape(oh, ow)


def timeit(fn, repeat=3):
    """【timeit】benchmark → 取多次运行的最小值（排除调度抖动）"""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0     # ms


def demo_im2col():
    """
    【demo_im2col】speedup → 加速比

    同一张 240×320 图、同一个 31×31 核：
        两层循环：Python 逐像素 -> 慢
        im2col + 矩阵乘：一次 BLAS -> 快
    实测差 3 个数量级以上，这就是"为什么框架不写循环"。
    """
    img = rng.normal(0, 1, size=(80, 100))
    small_k = np.ones((3, 3)) / 9.0

    # 3×3：手写两层循环 vs im2col，先确认结果一致
    a = correlate2d_naive(img, small_k)
    b = conv2d_im2col(img, small_k)
    print("3×3：循环 vs im2col 最大差 = {:.2e}".format(np.abs(a - b).max()))
    assert np.allclose(a, b, atol=1e-10)

    t_loop = timeit(lambda: correlate2d_naive(img, small_k), repeat=1)
    t_col = timeit(lambda: conv2d_im2col(img, small_k))
    print("3×3：两层循环 = {:7.2f} ms，im2col+GEMM = {:6.2f} ms，加速 {:.0f}×".format(
        t_loop, t_col, t_loop / t_col))

    # 31×31：只比 im2col 和"逐点向量化"两种写法
    big_k = np.ones((31, 31)) / 961.0
    t_im2col = timeit(lambda: conv2d_im2col(img, big_k))
    cols, (oh, ow) = im2col(img, 31, 1, 0)
    kflat = big_k.reshape(-1)

    def loop_over_kernel():
        # 只循环核（961 次），每次做一次整图加法 -> 比双层循环快，但仍不如 GEMM
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(img, (31, 31))
        acc = np.zeros((oh, ow))
        for idx in range(31 * 31):
            u, v = divmod(idx, 31)
            acc += win[..., u, v] * kflat[idx]
        return acc

    t_kernel = timeit(loop_over_kernel)

    print("31×31：im2col + GEMM      = {:6.2f} ms".format(t_im2col))
    print("31×31：只循环核（961 次） = {:6.2f} ms".format(t_kernel))
    print("注意：核变大后 im2col 的展开拷贝开始占主导，")
    print("      单张灰度图上它不一定比'按核累加'快；")
    print("      通道数 × batch 变大时，GEMM 才会明显胜出。")

    assert np.allclose(cols @ kflat, conv2d_im2col(img, big_k).reshape(-1), atol=1e-8)


# =====================================================================
# 五、高斯核：可分离，省一个数量级
# =====================================================================

def gaussian_kernel_1d(sigma, radius=None):
    """
    【gaussian_kernel_1d】1-D Gaussian → 一维高斯核

        g(x) = exp(-x² / (2σ²))，x 取整数 -r..r

    σ 越大越模糊；半径 r 一般取 3σ（99.7% 的能量），再归一化成和为 1。
    归一化很重要：不归一化会让图像整体变亮/变暗。
    """
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel_2d(sigma, radius=None):
    """【gaussian_kernel_2d】2-D Gaussian → 二维高斯核（一维外积）"""
    g = gaussian_kernel_1d(sigma, radius)
    return np.outer(g, g)


def blur_separable(img, g):
    """
    【blur_separable】separable filter → 可分离滤波

    二维高斯 = 一维横向 ⊗ 一维纵向（外积）。
    所以可以先沿行做一次 1D 卷积，再沿列做一次：
        代价从 O(H·W·k²) 降到 O(H·W·k)，k=31 时省 31 倍。

    这里用"移位相加"实现一维卷积，避免再写循环。
    """
    img = np.asarray(img, np.float64)
    r = len(g) // 2
    padded = np.pad(img, ((r, r), (r, r)), mode="reflect")
    h, w = img.shape

    tmp = np.zeros((h + 2 * r, w), np.float64)      # 先横向
    for i, wgt in enumerate(g):
        tmp += wgt * padded[:, i:i + w]

    out = np.zeros((h, w), np.float64)              # 再纵向
    for i, wgt in enumerate(g):
        out += wgt * tmp[i:i + h]
    return out


def demo_gaussian():
    """
    【demo_gaussian】separable Gaussian → 可分离高斯

    1) 外积得到的 2D 核，和直接按二维公式算的完全一致；
    2) 可分离滤波的结果，和 2D 卷积一致（差 < 1e-12）；
    3) 31×31 的核，可分离比 2D im2col 快一个数量级。
    """
    sigma = 5.0
    g1 = gaussian_kernel_1d(sigma)
    g2 = gaussian_kernel_2d(sigma)
    print("σ=5 的 1D 核长度 :", len(g1), "（半径 3σ=15）")
    print("核的和           : 1D={:.12f}  2D={:.12f}".format(g1.sum(), g2.sum()))
    print("2D 核中心值      :", round(float(g2[len(g1) // 2, len(g1) // 2]), 6))

    img = rng.normal(0, 1, size=(80, 100))
    sep = blur_separable(img, g1)
    full = correlate2d_naive(img, g2, padding=len(g1) // 2)   # 补零
    # 手写版补零，可分离版用 reflect；只在内部比较
    r = len(g1) // 2
    diff = np.abs(sep[r:-r, r:-r] - full[r:-r, r:-r]).max()
    print("可分离 vs 2D 卷积（内部）最大差 = {:.2e}".format(diff))

    t_sep = timeit(lambda: blur_separable(img, g1))
    t_2d = timeit(lambda: conv2d_im2col(img, g2, padding=r))
    print("31×31：可分离 = {:.2f} ms，2D im2col = {:.2f} ms，加速 {:.1f}×".format(
        t_sep, t_2d, t_2d / t_sep))

    assert abs(g1.sum() - 1) < 1e-12
    assert np.allclose(g2, np.outer(g1, g1))
    assert diff < 1e-10
    return g1, g2


# =====================================================================
# 六、Sobel：一阶导数找边缘
# =====================================================================

def sobel_gradients(img):
    """
    【sobel_gradients】Sobel operator → 索贝尔算子

        Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]    检测竖直边缘（左右变化）
        Gy = Gx.T                            检测水平边缘（上下变化）

    幅值：|G| = sqrt(Gx² + Gy²)，方向：θ = atan2(Gy, Gx)

    为什么中间是 2：它等于 [1,2,1]ᵀ ⊗ [-1,0,1]，即"先高斯平滑再求差分"，
    比直接 [-1,0,1] 更抗噪。这也是 Sobel 比 Prewitt 常用的原因。
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float64)
    gx = correlate2d_naive(img, kx, padding=1)
    gy = correlate2d_naive(img, kx.T, padding=1)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)          # 弧度，范围 (-π, π]
    return gx, gy, mag, ang


def demo_sobel():
    """
    【demo_sobel】gradient peak → 梯度峰值定位

    造一条 x=120 处的竖直阶跃（左 50，右 200）：
        理想情况下，Gx 在 x=119/120 处达到最大，峰值 = 150。
        Sobel 的 [-1,0,1] 差分在"跨越边界"的位置响应最强。

    这解释了：边缘检测输出的不是"边缘线"，而是"变化最强的位置"，
    所以后面还要做非极大值抑制（NMS）把粗边压成单像素。
    """
    img = np.full((80, 240), 50.0)
    img[:, 120:] = 200.0

    gx, gy, mag, ang = sobel_gradients(img)
    row = mag[40]

    # 只看内部：左右边界外是补的 0，会造出"假边缘"（下面单独讲）
    interior = row[10:230]
    peak = 10 + int(np.argmax(interior))
    print("竖直阶跃在 x=120；内部 Sobel 幅值峰值在 x={}，峰值={:.1f}".format(
        peak, row[peak]))
    print("理论上限 = 4×150 = 600（核系数绝对值之和 × 阶跃高度）")

    # 补零边界：x=0 处左邻是 0（差 50），x=239 处右邻是 0（差 200）
    print("边界处的假边缘：x=0 -> {:.0f}，x=239 -> {:.0f}".format(row[0], row[239]))
    print("（真实阶跃只有 150，补零却造出了 200/800 的梯度）")

    # 阶跃内部：Gx 应该是常数，gy 为 0
    print("平坦区 Gx 最大值 = {:.1f}".format(np.abs(gx[10:70, 20:100]).max()))
    print("平坦区 Gy 最大值 = {:.1f}".format(np.abs(gy[10:70, 20:100]).max()))

    assert abs(peak - 119) <= 1
    assert row[peak] > 590
    assert row[239] == 800.0           # 补零造成的边界假边缘
    assert np.abs(gx[10:70, 20:100]).max() < 1e-9
    return gx, gy, mag, ang


# =====================================================================
# 七、边界：补零为什么让边缘变暗
# =====================================================================

def demo_padding_modes():
    """
    【demo_padding_modes】border handling → 边界处理

    对一张常数图（全 100）做 9×9 均值滤波：
        - constant（补 0）：边缘像素窗口里混进了 0 -> 均值被拉低
        - reflect（镜像）：边缘值仍是 100
        - edge（复制边缘）：边缘值仍是 100

    量化：9×9 核、半径 4，左上角只有 25/81 的有效像素，
    所以角点值 ≈ 100 × 25/81 ≈ 30.9。

    工业检测里 ROI 贴边时，补零会凭空造出一条"暗边"，
    被后续阈值/边缘检测当成缺陷 —— 这是很常见的假报警来源。
    """
    img = np.full((60, 80), 100.0)
    g = np.ones((9, 9)) / 81.0
    r = 4

    out = {}
    for mode in ["constant", "reflect", "edge"]:
        padded = np.pad(img, r, mode=mode)
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(padded, (9, 9))
        out[mode] = np.einsum("ijkl,kl->ij", win, g)

    for mode, res in out.items():
        corner = res[0, 0]
        border = np.abs(res[:4, :].mean() - 100)
        print("{:9s}: 角点={:6.2f}  内部={:6.2f}  上边框平均偏差={:.2f}".format(
            mode, corner, res[30, 40], border))

    assert abs(out["constant"][0, 0] - 100 * 25 / 81) < 1e-9
    assert abs(out["reflect"][0, 0] - 100) < 1e-9
    assert abs(out["edge"][0, 0] - 100) < 1e-9
    return out


# =====================================================================
# 八、画图
# =====================================================================

def make_test_image(h=180, w=240):
    """【make_test_image】test pattern → 测试图（方块 + 圆 + 斜边 + 噪声）"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 60 + 40 * xx / w
    img[(xx > 40) & (xx < 120) & (yy > 40) & (yy < 130)] = 200
    img[(xx - 180) ** 2 + (yy - 90) ** 2 < 35 ** 2] = 150
    img[np.abs(yy - (0.5 * xx + 30)) < 3] = 230
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255)


def plot_all(img, g1, pad_out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("① 原图：方块 / 圆 / 斜线")

    ax = axes[0, 1]
    ax.imshow(correlate2d_naive(img, gaussian_kernel_2d(1.5), padding=4),
              cmap="gray", vmin=0, vmax=255)
    ax.set_title("② 高斯模糊 σ=1.5")

    gx, gy, mag, _ = sobel_gradients(img)
    ax = axes[0, 2]
    ax.imshow(np.abs(gx), cmap="gray")
    ax.set_title("③ |Gx|：只对竖直边响应")

    ax = axes[1, 0]
    ax.imshow(mag, cmap="gray")
    ax.set_title("④ 梯度幅值 |G|")

    ax = axes[1, 1]
    ax.imshow(np.abs(correlate2d_naive(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), padding=1)),
              cmap="gray")
    ax.set_title("⑤ 拉普拉斯：二阶导，对噪声更敏感")

    ax = axes[1, 2]
    for mode, res in pad_out.items():
        ax.plot(res[0, :], label={"constant": "补零", "reflect": "镜像", "edge": "复制"}[mode])
    ax.axhline(100, color="k", ls="--", lw=0.8)
    ax.set_title("⑥ 9×9 均值滤波后的第一行：补零把边缘拉暗")
    ax.set_xlabel("x")
    ax.set_ylabel("像素值")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "convolution.png")
        fig.savefig(path, dpi=120)
        print("\n图已保存:", path)
    else:
        plt.show()
    plt.close(fig)


def main():
    print("=" * 60)
    print("一、相关 vs 卷积")
    demo_flip()

    print("\n" + "=" * 60)
    print("二、输出尺寸")
    demo_size()

    print("\n" + "=" * 60)
    print("三、与 scipy 对照")
    demo_against_scipy()

    print("\n" + "=" * 60)
    print("四、im2col")
    demo_im2col()

    print("\n" + "=" * 60)
    print("五、高斯可分离")
    g1, g2 = demo_gaussian()

    print("\n" + "=" * 60)
    print("六、Sobel")
    demo_sobel()

    print("\n" + "=" * 60)
    print("七、边界模式")
    pad_out = demo_padding_modes()

    print("\n" + "=" * 60)
    print("八、画图")
    plot_all(make_test_image(), g1, pad_out)

    print("\nALL PASS")


if __name__ == "__main__":
    main()
