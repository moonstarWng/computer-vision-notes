# -*- coding: utf-8 -*-
"""
M2 第 2 篇 —— 畸变与去畸变

对应笔记：notes/M2-相机模型与标定/2-畸变与去畸变.md

这个脚本回答五个问题
--------------------
1. 畸变模型长什么样？径向和切向各管什么？
2. 桶形和枕形差在哪？一个参数（k1 的符号）就能看出来？
3. 为什么"去畸变"必须迭代？正向有闭式解，反向没有。
4. 把顺序写错（先乘 K 再加畸变）会差多少像素？
5. 去畸变之后，原来的 K 还能直接用吗？视场会怎么变？

运行：
    python 02_distortion.py
预期：最后打印 ALL PASS。
只依赖 numpy。
"""

import numpy as np

W, H = 1280, 1024
FX, FY = 2318.84, 2318.84
CX, CY = 639.5, 511.5

# 一组典型的中等广角镜头参数（数值取自一次真实标定的量级）
K1, K2, P1, P2 = -0.28, 0.07, 0.001, -0.0015


# =====================================================================
# 一、畸变模型：归一化坐标上加两个多项式
# =====================================================================

def distort_normalized(xn, yn, k1=K1, k2=K2, p1=P1, p2=P2):
    """
    【distort_normalized】distort → 理想归一化坐标 -> 畸变后的归一化坐标

        r² = x² + y²
        径向：x_d = x·(1 + k1·r² + k2·r⁴)          ← 只跟距离有关
        切向：x_d += 2·p1·x·y + p2·(r² + 2x²)      ← 跟方位有关，镜片没装正

    为什么在归一化坐标上做：畸变是**镜头+装配**的属性，与像素分辨率无关。
    在归一化坐标里标定出来的 k1 换一台同镜头相机仍然可用。
    """
    xn = np.asarray(xn, dtype=np.float64)
    yn = np.asarray(yn, dtype=np.float64)
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def undistort_normalized(xd, yd, k1=K1, k2=K2, p1=P1, p2=P2, iters=10):
    """
    【undistort_normalized】undistort → 畸变坐标 -> 理想归一化坐标

    正向（理想 -> 畸变）有闭式解，反向没有，所以用不动点迭代：

        x ← (x_d − 切向项(x)) / 径向因子(x)

    每次迭代把"当前猜测"代进去重算一次修正量。k 越小收敛越快；
    广角镜头（|k1| 大）在图像边缘可能要 10~20 次。
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - tx) / radial
        y = (yd - ty) / radial
    return x, y


def to_pixel(xn, yn, fx=FX, fy=FY, cx=CX, cy=CY):
    """【to_pixel】normalized -> pixel（这就是 K 的作用）"""
    return xn * fx + cx, yn * fy + cy


def to_normalized(u, v, fx=FX, fy=FY, cx=CX, cy=CY):
    """【to_normalized】pixel -> normalized（K⁻¹ 的作用）"""
    return (u - cx) / fx, (v - cy) / fy


# =====================================================================
# 二、桶形 / 枕形：只看 k1 的符号
# =====================================================================

def line_sag(k1, y0=0.22, k2=0.0, n=201):
    """
    【line_sag】line sag → 一条水平直线被畸变弯了多少（归一化单位）

    取图像中间偏上的一条水平线 y = y0，看它畸变后偏离直线的最大量。
    k1 < 0（桶形）时，越靠边（r 越大）压缩越多，线的两端被"拉回来"，
    于是中间显得更靠上——看起来就是向外鼓。
    """
    x = np.linspace(-0.28, 0.28, n)
    y = np.full_like(x, y0)
    xd, yd = distort_normalized(x, y, k1=k1, k2=k2, p1=0.0, p2=0.0)
    # 用两端点连一条直线，量中间偏离多少（带符号：>0 表示向外鼓）
    slope = (yd[-1] - yd[0]) / (xd[-1] - xd[0])
    line = yd[0] + slope * (xd - xd[0])
    return float((yd - line)[n // 2])


# =====================================================================
# 三、demo
# =====================================================================

def demo_corner_shift():
    print("一、角点被拉走多少（像素）")
    print("  参数: k1={:+.2f}, k2={:+.2f}, p1={:+.4f}, p2={:+.4f}".format(K1, K2, P1, P2))
    corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]
    for u, v in corners:
        xn, yn = to_normalized(u, v)
        xd, yd = distort_normalized(xn, yn)
        ud, vd = to_pixel(xd, yd)
        d = np.hypot(ud - u, vd - v)
        r = np.hypot(xn, yn)
        print("  角点 ({:4d},{:4d}): r={:.4f}  理想->畸变 位移 {:.2f} px".format(u, v, r, d))

    # 径向因子表：只看距离
    print("\n  归一化半径 r -> 径向因子 (1 + k1*r^2 + k2*r^4)")
    for r in (0.05, 0.10, 0.20, 0.30, 0.35):
        f = 1 + K1 * r * r + K2 * r ** 4
        print("    r={:.2f} -> {:.4f}  (差 {:+.2f}%)".format(r, f, (f - 1) * 100))
    return corners


def demo_barrel_pincushion():
    print("\n二、桶形 vs 枕形：只改 k1 的符号")
    for k1, name in ((-0.28, "桶形 k1<0"), (+0.28, "枕形 k1>0")):
        sag = line_sag(k1)
        # 角点位移方向：向外还是向内
        xn, yn = to_normalized(W - 1, H - 1)
        xd, yd = distort_normalized(xn, yn, k1=k1, k2=0.0, p1=0.0, p2=0.0)
        dr = np.hypot(xd, yd) - np.hypot(xn, yn)
        print("  {}: 角点径向变化 {:+.4f}（{}），直线弯曲 {:.2f} px".format(
            name, dr, "向外" if dr > 0 else "向内", sag * FX))
    print("  读法：桶形把边角向内压（直线向外鼓），枕形把边角向外推（直线向内凹）")
    assert line_sag(-0.28) > 0
    assert line_sag(+0.28) < 0


def demo_undistort_convergence():
    print("\n三、去畸变：为什么必须迭代")
    # 取一个靠近边角的点，正向后反解，看残差随迭代次数下降
    xn, yn = to_normalized(1200, 950)
    xd, yd = distort_normalized(xn, yn)
    print("  目标点 理想({:.6f},{:.6f}) -> 畸变({:.6f},{:.6f})".format(xn, yn, xd, yd))
    for it in (1, 2, 3, 5, 10):
        xr, yr = undistort_normalized(xd, yd, iters=it)
        err = max(abs(xr - xn), abs(yr - yn)) * FX
        print("    迭代 {:2d} 次 -> 残余 {:.3e} px".format(it, err))
    xr, yr = undistort_normalized(xd, yd, iters=20)
    err = max(abs(xr - xn), abs(yr - yn)) * FX
    assert err < 1e-9
    print("  结论：梯度越大的边缘越慢；20 次后到浮点精度")


def demo_wrong_order():
    print("\n四、顺序写错会差多少")
    xn, yn = to_normalized(1200, 950)
    xd, yd = distort_normalized(xn, yn)
    u_ok, v_ok = to_pixel(xd, yd)

    # 错误写法：把归一化的修正量当成像素修正量，直接加到像素上（忘了乘 fx）
    r2 = xn * xn + yn * yn
    du = K1 * r2 * xn
    dv = K1 * r2 * yn
    u_bad, v_bad = xn * FX + CX + du, yn * FY + CY + dv

    print("  正确: 先畸变再乘 K -> ({:.3f}, {:.3f})".format(u_ok, v_ok))
    print("  错误: 先乘 K 再加修正（修正量没乘 fx） -> ({:.3f}, {:.3f})".format(u_bad, v_bad))
    print("  差 {:.2f} px —— 修正量是归一化单位，加到像素坐标上前必须乘 fx".format(
        np.hypot(u_bad - u_ok, v_bad - v_ok)))
    assert np.hypot(u_bad - u_ok, v_bad - v_ok) > 10


def demo_fov_change():
    print("\n五、去畸变之后，视场变了")
    # 畸变图的四个角：反解到理想坐标，看它们落到哪里
    corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]
    us, vs = [], []
    for u, v in corners:
        xd, yd = to_normalized(u, v)
        x, y = undistort_normalized(xd, yd)
        uu, vv = to_pixel(x, y)
        us.append(uu)
        vs.append(vv)
        print("  畸变图角点 ({:4d},{:4d}) -> 去畸变后 ({:8.2f},{:8.2f})".format(u, v, uu, vv))

    span_u = max(us) - min(us)
    span_v = max(vs) - min(vs)
    scale = max(span_u / (W - 1), span_v / (H - 1))
    print("  理想图跨度 {:.1f} x {:.1f} px（原图 {} x {}）".format(span_u, span_v, W - 1, H - 1))
    print("  要装下全部内容，需放大约 {:.4f} 倍，或裁掉四周黑边".format(scale))
    print("  OpenCV 对应 getOptimalNewCameraMatrix(alpha=0 裁掉 / alpha=1 保留黑边)")
    print("  注意：裁掉之后 K 也变了（fx、cx、cy 都乘了 scale）")
    assert scale > 1.0


def main():
    demo_corner_shift()
    demo_barrel_pincushion()
    demo_undistort_convergence()
    demo_wrong_order()
    demo_fov_change()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
