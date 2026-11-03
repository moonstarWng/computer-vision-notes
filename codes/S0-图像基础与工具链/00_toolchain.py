# -*- coding: utf-8 -*-
"""
S0 第 0 篇 —— NumPy 工具链：数组、轴、dtype、视图、坐标

对应笔记：notes/S0-图像基础与工具链/0-环境与工具链.md

这个脚本回答四个问题
--------------------
1. 一张图的 shape 是 (H, W, 3)，那 axis=0/1/2 分别是什么？
2. 为什么 uint8 相加会"绕回去"？什么时候必须转 float32？
3. 切片改图会改到原图，什么时候是坑、什么时候是特性？
4. 图像坐标 (x, y) 和数组下标 (row, col) 怎么互转？

运行：
    python 00_toolchain.py
预期：最后打印 ALL PASS；任何断言失败会直接报错。
"""

import numpy as np

rng = np.random.default_rng(0)


# =====================================================================
# 一、shape 与 axis：先分清"高、宽、通道"
# =====================================================================

def demo_shape_axis():
    """
    【demo_shape_axis】shape / axis → 形状与轴

    一张彩色图是三维数组：
        shape = (H, W, 3) = (高, 宽, 通道)
        img[r, c, k]  = 第 r 行、第 c 列、第 k 个通道的像素值

    axis 的规则只有一句：
        axis=i 表示"沿第 i 维变化"，运算结束后这一维会消失（或被保留成 1）。
    """
    H, W = 4, 6
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    print("shape        :", img.shape)          # (4, 6, 3)
    print("ndim / size  :", img.ndim, img.size)  # 3 / 72
    print("每像素字节数 :", img.itemsize, "总字节:", img.nbytes)

    # axis=0：沿"行"方向压缩 → 得到每一列的均值，形状 (W, 3)
    col_mean = img.mean(axis=0)
    # axis=1：沿"列"方向压缩 → 得到每一行的均值，形状 (H, 3)
    row_mean = img.mean(axis=1)
    # axis=2：沿"通道"方向压缩 → 得到灰度图，形状 (H, W)
    gray = img.mean(axis=2)

    print("mean(axis=0) :", col_mean.shape, " <- 每个 x 位置一条竖线的均值")
    print("mean(axis=1) :", row_mean.shape, " <- 每个 y 位置一条横线的均值")
    print("mean(axis=2) :", gray.shape, " <- 三通道取平均，变成灰度图")

    assert col_mean.shape == (W, 3)
    assert row_mean.shape == (H, 3)
    assert gray.shape == (H, W)

    # 下面这种写法是新手最常见的 bug：
    #   img.mean(axis=2) 才是灰度；写成 axis=0 得到的是 (W,3)，不是图。
    return img


# =====================================================================
# 二、dtype：uint8 会溢出，float32 才有小数
# =====================================================================

def demo_dtype_overflow():
    """
    【demo_dtype_overflow】dtype overflow → 整型溢出

    uint8 的范围是 [0, 255]，共 2^8 个值。
    超出范围时 NumPy 不会报错，而是"模 256"绕回去：
        200 + 100 = 300 -> 44
        10  - 20  = -10 -> 246

    这就是为什么"两帧相减"必须先把 dtype 转成 float32，
    否则亮处减暗处会突然变成 246（白），而不是负数。
    """
    a = np.uint8(200)
    b = np.uint8(100)
    # overflow="ignore" 只是为了不让 NumPy 打警告；溢出本身照样发生
    with np.errstate(over="ignore"):
        s = a + b
        d = np.uint8(10) - np.uint8(20)
    print("uint8: 200 + 100 =", s, "（期望 300，实际绕回 44）")
    print("uint8: 10  - 20  =", d, "（期望 -10，实际 246）")

    # 正确做法：先转 float32 再算，最后再决定要不要截断回 uint8
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    d32 = np.float32(10) - np.float32(20)
    print("float32: 200 + 100 =", a32 + b32)
    print("float32: 10  - 20  =", d32)

    # 浮点图像还有一种约定：把 [0,255] 归一化到 [0,1] 再处理
    img_u8 = rng.integers(0, 256, size=(3, 3), dtype=np.uint8)
    img_f = img_u8.astype(np.float32) / 255.0
    print("归一化后范围 : [{:.3f}, {:.3f}]".format(img_f.min(), img_f.max()))

    assert s == 44
    assert d == 246
    assert a32 + b32 == 300.0
    assert d32 == -10.0


# =====================================================================
# 三、广播：不用复制内存，也能"每个通道减自己的均值"
# =====================================================================

def demo_broadcast():
    """
    【demo_broadcast】broadcasting → 广播

    规则：从最后一维往前比，维度相等或其中一个为 1 就能广播。

        (H, W, 3)  -  (3,)      -> (H, W, 3)   每个通道减自己的均值
        (H, W, 1)  -  (H, W)    -> 报错！最后一维 3 vs W 不匹配
        (H, W)     -  (H, 1)    -> (H, W)      每一行减自己的均值

    为什么关心：白平衡、逐通道归一化、减去背景，都是这个形状。
    """
    img = rng.normal(120, 10, size=(4, 6, 3)).astype(np.float32)

    per_channel = img.mean(axis=(0, 1))      # (3,) 三个通道各自的均值
    balanced = img - per_channel             # (4,6,3) - (3,) 广播

    print("每通道均值   :", np.round(per_channel, 2))
    print("减去后均值   :", np.round(balanced.mean(axis=(0, 1)), 6), "（应接近 0）")

    # 保持维度：keepdims=True 得到 (1,1,3)，广播时更直观（也不会写错轴）
    mean_keep = img.mean(axis=(0, 1), keepdims=True)
    print("keepdims 形状:", mean_keep.shape)

    # float32 求和的累积误差约 1e-5 量级，容差不能比它还小
    assert np.allclose(balanced.mean(axis=(0, 1)), 0, atol=1e-4)
    assert mean_keep.shape == (1, 1, 3)


# =====================================================================
# 四、视图 vs 拷贝：切片是"窗户"，不是"复印件"
# =====================================================================

def demo_view_copy():
    """
    【demo_view_copy】view / copy → 视图与拷贝

    img[10:20, 10:20] 返回的是原数组的一个"视图"（view）：
        改它，原图跟着变；它不占新内存。

    需要独立副本时显式 .copy()。

    什么时候是特性：ROI 原地修改（画框、抠图、inpaint）。
    什么时候是坑：把切片存起来当"备份"，结果原图一改备份也变了。
    """
    img = np.zeros((6, 6), dtype=np.uint8)

    roi = img[1:4, 2:5]        # 视图
    roi[:] = 255               # 改视图 = 改原图
    print("改视图后原图非零像素数:", int((img > 0).sum()), "（期望 9）")

    roi_copy = img[1:4, 2:5].copy()
    roi_copy[:] = 7            # 改副本，原图不变
    print("改副本后原图 (2,3) 的值:", int(img[2, 3]), "（仍是 255）")

    # 判断是不是视图：.base 指向原数组就说明是视图
    print("roi.base is img      :", roi.base is img)
    print("roi_copy.base is img :", roi_copy.base is img)

    assert (img > 0).sum() == 9
    assert img[2, 3] == 255
    assert roi_copy.base is None


# =====================================================================
# 五、坐标：图像 (x, y) 与数组 (row, col) 是反的
# =====================================================================

def demo_coordinate_roundtrip():
    """
    【demo_coordinate_roundtrip】coordinate convention → 坐标约定

    数学/几何/标定里说"点 (x, y)"：
        x 向右，y 向下（图像坐标系），单位像素。

    NumPy 下标说"img[row, col]"：
        row 是行号（从上往下），col 是列号（从左往右）。

    对应关系：
        row = y
        col = x
        img[y, x] 才是"第 y 行第 x 列"

    画图函数（matplotlib）用的是 (x, y)，所以 plt.plot(x, y) 里
    两个参数顺序和 img[y, x] 正好相反 —— 这是 90% 的"图转 90 度"来源。
    """
    H, W = 100, 200
    x, y = 30, 70                      # 图像坐标：向右 30，向下 70
    row, col = y, x

    img = np.zeros((H, W), dtype=np.uint8)
    img[row, col] = 255

    ys, xs = np.nonzero(img)           # nonzero 返回 (row, col)，即 (y, x)
    print("写入 (x=30, y=70) -> 下标 (row=70, col=30)")
    print("nonzero 返回 (ys, xs):", ys.tolist(), xs.tolist())

    assert (ys[0], xs[0]) == (y, x)

    # 常见错误：img[x, y] 会把点画到 (row=30, col=70)，看起来"沿对角线翻了"
    wrong = np.zeros((H, W), dtype=np.uint8)
    wrong[x, y] = 255
    wy, wx = np.nonzero(wrong)
    print("若写成 img[x, y] 实际落在 (x={}, y={})".format(int(wx[0]), int(wy[0])))
    assert (wy[0], wx[0]) == (x, y)    # 注意这里是 (y, x) = (30, 70)，错了


# =====================================================================
# 六、assert：把"我以为"变成"跑得过"
# =====================================================================

def demo_assert():
    """
    【demo_assert】assertion → 断言

    学习阶段最值钱的一句话：不要用 print 看数，用 assert 锁结论。

        assert abs(a - b) < 1e-6, "a 和 b 应该相等"

    浮点比较永远带容差：
        np.allclose(a, b, atol=1e-6)     绝对误差
        np.allclose(a, b, rtol=1e-6)     相对误差

    断言失败会抛 AssertionError，比"结果看起来差不多"可靠得多。
    """
    x = 0.1 + 0.2
    print("0.1 + 0.2 =", repr(x), " 等于 0.3 吗:", x == 0.3)
    assert not (x == 0.3)
    assert np.isclose(x, 0.3)

    # 反例：不要写 assert (a == b) 去比较浮点数组
    a = np.array([1.0, 2.0, 3.0])
    b = a * 1.0000001
    assert np.allclose(a, b, rtol=1e-6)
    print("浮点比较：直接 == 不可靠，用 np.allclose")


def main():
    print("=" * 60)
    print("一、shape 与 axis")
    demo_shape_axis()

    print("\n" + "=" * 60)
    print("二、dtype 与溢出")
    demo_dtype_overflow()

    print("\n" + "=" * 60)
    print("三、广播")
    demo_broadcast()

    print("\n" + "=" * 60)
    print("四、视图 vs 拷贝")
    demo_view_copy()

    print("\n" + "=" * 60)
    print("五、坐标约定")
    demo_coordinate_roundtrip()

    print("\n" + "=" * 60)
    print("六、assert")
    demo_assert()

    print("\nALL PASS")


if __name__ == "__main__":
    main()
