# -*- coding: utf-8 -*-
"""
M2 第 1 篇 —— 针孔模型与内参

对应笔记：notes/M2-相机模型与标定/1-针孔模型与内参.md

这个脚本回答四个问题
--------------------
1. 内参 fx 到底是什么？怎么从物理焦距和像元尺寸算出来？
2. 3D 点怎么变成像素？投影一次要几个矩阵、各自负责什么？
3. 从像素能反推 3D 点吗？为什么不能、缺的是什么？
4. 图像 resize / crop 之后，内参该怎么改？

运行：
    python 01_pinhole.py
预期：最后打印 ALL PASS。
只依赖 numpy，不出图（本篇的重点是坐标系和单位，不是可视化）。
"""

import numpy as np


# =====================================================================
# 一、内参：从毫米到像素
# =====================================================================

def intrinsic(f_mm, pixel_size_um, width, height, cx=None, cy=None):
    """
    【intrinsic】intrinsic matrix → 内参矩阵 K

    针孔模型里，3D 点先除以深度变成"归一化坐标"，再乘焦距变成像素：

        u = fx · (X/Z) + cx
        v = fy · (Y/Z) + cy

    把这两行写成齐次矩阵，就是经典的 K：

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    fx 的单位是**像素**，不是毫米：
        fx = f / dx
    其中 f 是物理焦距（mm），dx 是一个像元的宽度（mm）。
    这就是"像素焦距"的来历——它把"毫米"和"像素"两个单位接在一起。

    参数：
        f_mm        ：物理焦距，mm
        pixel_size_um：像元尺寸，微米（工业相机常见 2.2 / 3.45 / 5.5）
        width/height：图像分辨率，像素
        cx/cy       ：主点，默认取图像中心（真实相机不在正中心，见 §3）
    """
    dx_mm = pixel_size_um * 1e-3          # 微米 -> 毫米
    fx = f_mm / dx_mm
    fy = fx                                # 方形像元时 fy = fx
    cx = (width - 1) / 2.0 if cx is None else cx
    cy = (height - 1) / 2.0 if cy is None else cy
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    return K


def pixel_size_mm(f_mm, fx):
    """【pixel_size_mm】pixel size → 由 fx 反推像元尺寸（mm）。"""
    return f_mm / fx


def mm_per_pixel(K, Z_mm):
    """
    【mm_per_pixel】millimeter per pixel → 某个深度上，一个像素代表多少毫米

        mm/px = Z / fx

    这是单目测量最核心的一行：**同样的相机，越远越粗**。
    注意它和双目不同：双目是 Z² 关系（见 M2 双目篇），单目是线性。
    """
    fx = K[0, 0]
    return Z_mm / fx


# =====================================================================
# 二、投影：世界 → 相机 → 归一化平面 → 像素
# =====================================================================

def rodrigues(w):
    """
    【rodrigues】axis-angle → rotation matrix（罗德里格斯公式）

    w 是旋转向量（弧度），|w| 是旋转角，w/|w| 是转轴。
    这里手写一遍，是为了让"外参 R 从哪来"这件事不留黑箱。
    """
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    Kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)


def project(K, R, t, X):
    """
    【project】project → 世界点投影到像素

        X_cam = R · X_world + t          （外参：把世界搬到相机坐标系）
        x_n   = X_cam[:2] / X_cam[2]     （透视除法：除以深度）
        uv    = K · [x_n, 1]             （内参：归一化平面 -> 像素）

    参数：
        X：(N, 3) 世界点，mm
        R, t：世界 -> 相机的外参（R 是相机看世界的旋转，t 是平移，mm）
    返回：
        uv：(N, 2) 像素坐标；depth：(N,) 每个点在相机系的 Z
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xc = X @ R.T + t.reshape(1, 3)
    depth = Xc[:, 2]
    if np.any(depth <= 1e-9):
        raise ValueError("有点落在相机背后或正好在光心（Z <= 0），投影无意义")
    xn = Xc[:, :2] / depth[:, None]          # 归一化坐标（无单位，等于 tan(角度)）
    uv = xn @ K[:2, :2].T + K[:2, 2]          # 等价于 K @ [xn,1]
    return uv, depth


def backproject_direction(K, uv):
    """
    【backproject_direction】backproject → 像素反投影成一条射线

        d = K⁻¹ · [u, v, 1]     方向，长度归一化到 z=1

    关键：反投影得到的**只有方向，没有距离**。
    像素丢掉了深度——这是单目无法直接测距的根本原因。
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    K_inv = np.linalg.inv(K)
    homo = np.hstack([uv, np.ones((uv.shape[0], 1))])
    d = homo @ K_inv.T
    return d / d[:, 2:3]


# =====================================================================
# 三、改图像尺寸：K 必须跟着改
# =====================================================================

def scale_intrinsics(K, s):
    """
    【scale_intrinsics】resize → 图像缩放 s 倍后，K 怎么改

    缩放改的是"一个像素代表多大面积"，所以 fx、fy、cx、cy 全部乘 s。
        fx' = s·fx,  cx' = s·cx
    忘记改 K，就是"图像改了、相机参数没改"，重投影会整体偏移。
    """
    K2 = K.copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    return K2


def crop_intrinsics(K, x0, y0):
    """
    【crop_intrinsics】ROI 裁剪 → 左上角裁掉 (x0, y0) 后，K 怎么改

    裁剪只平移原点，不动焦距：fx、fy 不变，cx、cy 减掉偏移。
    做 ROI 测量时最容易漏这一步。
    """
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return K2


# =====================================================================
# 四、demo
# =====================================================================

def make_cube(edge=100.0, center=(0.0, 0.0, 0.0)):
    """【make_cube】cube → 边长 edge 的立方体 8 个角点（世界坐标，mm）

    默认放在世界原点；要用 t 把相机退到 500 mm 处，而不是把立方体挪到 Z=500，
    否则会得到"距离 1000 mm"的深度——这是新手最常犯的一次重复平移。
    """
    h = edge / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return signs * h + np.asarray(center, float)


def demo_intrinsic():
    print("一、内参：从毫米到像素")
    f_mm, px_um = 8.0, 3.45
    W, H = 1280, 1024
    K = intrinsic(f_mm, px_um, W, H)
    fx = K[0, 0]
    print("  焦距 f          = {:.2f} mm".format(f_mm))
    print("  像元尺寸 dx     = {:.2f} um = {:.5f} mm".format(px_um, px_um * 1e-3))
    print("  fx = f/dx       = {:.2f} px".format(fx))
    print("  主点 (cx, cy)   = ({:.1f}, {:.1f})".format(K[0, 2], K[1, 2]))

    # 物理尺寸与视场角
    sensor_w = W * px_um * 1e-3
    sensor_h = H * px_um * 1e-3
    fov_h = 2 * np.degrees(np.arctan(sensor_w / (2 * f_mm)))
    fov_v = 2 * np.degrees(np.arctan(sensor_h / (2 * f_mm)))
    print("  传感器尺寸      = {:.3f} x {:.3f} mm".format(sensor_w, sensor_h))
    print("  水平/垂直 FOV   = {:.2f}° / {:.2f}°".format(fov_h, fov_v))

    # 单位换算自检
    assert np.isclose(pixel_size_mm(f_mm, fx), px_um * 1e-3)
    assert np.isclose(fx * px_um * 1e-3, f_mm)
    return K, f_mm


def demo_mm_per_pixel(K):
    print("\n二、一个像素值多少毫米")
    print("  深度 Z(mm) | mm/px | 100mm 物体占多少像素")
    for Z in (300.0, 500.0, 1000.0, 2000.0):
        r = mm_per_pixel(K, Z)
        print("  {:9.0f} | {:5.3f} | {:6.1f}".format(Z, r, 100.0 / r))

    r500 = mm_per_pixel(K, 500.0)
    r1000 = mm_per_pixel(K, 1000.0)
    # 单目精度随距离线性变粗：距离翻倍，mm/px 也翻倍
    assert np.isclose(r1000 / r500, 2.0)
    return r500


def demo_project(K):
    print("\n三、投影：世界 → 相机 → 归一化平面 → 像素")
    X = make_cube(100.0)
    R = rodrigues(np.array([0.05, 0.10, 0.20]))   # 小角度姿态
    t = np.array([10.0, -5.0, 500.0])

    uv, depth = project(K, R, t, X)
    print("  立方体 8 个角点的深度范围: {:.1f} ~ {:.1f} mm".format(depth.min(), depth.max()))
    print("  前 3 个角点 (X,Y,Z) -> (u,v):")
    for i in range(3):
        print("    ({:7.2f},{:7.2f},{:7.2f}) -> ({:8.3f},{:8.3f})".format(
            X[i, 0], X[i, 1], X[i, 2], uv[i, 0], uv[i, 1]))

    # 投影的"逆"：从像素反投影得到的射线，必须穿过原来的 3D 点
    d = backproject_direction(K, uv)
    Xc = X @ R.T + t.reshape(1, 3)
    d_true = Xc / Xc[:, 2:3]
    err = np.abs(d - d_true).max()
    print("  反投影射线 vs 真实方向 最大差 = {:.2e}".format(err))

    # 像素是否落在图内
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 1024))
    print("  落在图内的角点数: {}/8".format(int(inside.sum())))

    assert err < 1e-12
    assert inside.all()
    return X, R, t, uv, depth


def demo_backproject(K):
    print("\n四、反投影：只有方向，没有距离")
    u, v = 900.0, 400.0
    d = backproject_direction(K, np.array([[u, v]]))[0]
    print("  像素 ({:.0f},{:.0f}) -> 射线方向 ({:.4f}, {:.4f}, 1)".format(u, v, d[0], d[1]))
    for Z in (500.0, 1000.0):
        P = d * Z
        u2, _ = project(K, np.eye(3), np.zeros(3), P.reshape(1, 3))
        print("    沿射线走 Z={:6.0f} mm -> 点 ({:8.2f},{:8.2f},{:8.2f})，再投影回 ({:.1f},{:.1f})".format(
            Z, P[0], P[1], P[2], u2[0, 0], u2[0, 1]))
    print("  同一条射线上的所有点都投影到同一个像素 —— 这就是单目缺深度的含义")


def demo_resize_crop(K):
    print("\n五、resize / crop 之后，K 必须跟着改")
    X = make_cube(100.0)
    R, t = np.eye(3), np.array([0.0, 0.0, 500.0])
    uv, _ = project(K, R, t, X)

    # 方式一：先把图缩小一半，再投影 -> 应该等于"原投影坐标 × 0.5"
    # 注意：OpenCV 的 resize 用的是像素中心约定，这里验证的是 K 的一致性
    K_half = scale_intrinsics(K, 0.5)
    uv_flag = project(K, R, t, X)[0]
    # 同一组 3D 点，用半尺寸 K 投影（世界不变），像素应减半
    uv_half = project(K_half, R, t, X)[0]
    print("  原图投影前 2 点: ", np.round(uv_flag[:2], 3).tolist())
    print("  半尺寸 K 投影  : ", np.round(uv_half[:2], 3).tolist())
    print("  是否等于原坐标的一半: ", bool(np.allclose(uv_half, uv_flag * 0.5)))

    # 方式二：ROI 裁掉左上角 (100, 50)，内参平移
    K_roi = crop_intrinsics(K, 100.0, 50.0)
    uv_roi = project(K_roi, R, t, X)[0]
    print("  裁掉 (100,50) 后投影偏移: ", np.round((uv_flag - uv_roi)[:2], 3).tolist())
    print("  偏移应恒为 (100, 50)")

    assert np.allclose(uv_half, uv_flag * 0.5)
    assert np.allclose(uv_flag - uv_roi, np.array([100.0, 50.0]))
    return K_half, K_roi


def main():
    K, f_mm = demo_intrinsic()
    demo_mm_per_pixel(K)
    demo_project(K)
    demo_backproject(K)
    demo_resize_crop(K)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
