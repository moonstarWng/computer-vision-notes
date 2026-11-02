# -*- coding: utf-8 -*-
"""
M1 第 1 篇 · 补充图示：把 SO(3) 画出来
对应笔记：notes/M1-数学与几何基础/1-旋转.md（§1.4 流形、§2.2 旋转向量）

这个脚本做四张图，帮助建立"旋转作用在 3D 空间里是什么样"的直觉：

    图 1  一个不对称的盒子被 R 旋转后，三根坐标轴跟着转；红虚线是转轴，
          橙色曲线是盒子的一个角点扫过的圆弧。
    图 2  绕同一根轴连续转动的"扫掠"：向量每隔 15° 画一次，连成一段圆弧。
    图 3  SO(3) 的"球"：半径 π 的实心球。球内一点 = 一个旋转；
          球面上的 ω 和 −ω 是同一个旋转（对径点粘合）。
    图 4  小角度时 exp(ω^) ≈ I + [ω]× 的误差曲线（为什么优化能用一阶近似）。

运行
----
    python 05_so3_visual.py            # 直接弹出窗口显示
    python 05_so3_visual.py --save     # 只保存成 out/so3_visual.png，不弹窗

说明：默认是**直接显示、不写文件**；只有加 --save 才会保存图片。
"""

import os
import sys

import numpy as np

import matplotlib
# 默认用交互后端（Windows 上通常是 TkAgg），直接弹窗；
# 加 --save 时才切成不弹窗的 Agg，把图写到文件。
SAVE = "--save" in sys.argv
if SAVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 让中文能正常显示（Windows 自带微软雅黑；没有就退回黑体/DejaVu）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 一、旋转相关的小工具（和 01_rotation.py 同一套公式）
# =====================================================================

def hat(w):
    """向量 -> 叉乘矩阵"""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def rodrigues(w):
    """旋转向量 ω = θk -> 旋转矩阵 R"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_axis_angle(R):
    """从 R 取回 (轴 k, 角度 θ)，只用于打印"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-9:
        return np.zeros(3), 0.0
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return k, theta


# =====================================================================
# 二、要"被转"的物体：一个三维尺寸各不相同的盒子（不对称，转起来明显）
# =====================================================================

SIZE = np.array([1.2, 0.6, 0.3]) / 2.0    # 半边长

CORNERS = np.array([[sx, sy, sz]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)]) * SIZE

# 12 条棱：任意两个顶点只差一个坐标
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.sum(np.abs(CORNERS[i] - CORNERS[j]) > 1e-9) == 1]


def box_edges(R=None, t=None):
    """返回盒子的 12 条棱（可选地先做旋转 R 和平移 t）"""
    P = CORNERS if R is None else CORNERS @ R.T
    if t is not None:
        P = P + t
    return [[P[i], P[j]] for i, j in EDGES]


def draw_axes(ax, R=None, length=1.0, lw=2.0, alpha=1.0, label=False):
    """画三根坐标轴（红=x，绿=y，蓝=z）；给 R 就画"转过去之后"的轴"""
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    names = ["x", "y", "z"]
    E = np.eye(3) if R is None else R
    for i in range(3):
        v = E[:, i] * length            # R 的第 i 列 = 原坐标轴转过去的方向
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i],
                  linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
        if label:
            ax.text(v[0] * 1.10, v[1] * 1.10, v[2] * 1.10, names[i],
                    color=colors[i], fontsize=10, weight="bold")


def style_3d(ax, lim=1.8, title="", zoom=1.45):
    """统一的 3D 坐标轴样式：等比例 + 放大 + 去掉多余文字"""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    # 三轴等比例，否则旋转会被拉伸变形；zoom 让画面填满子图
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(title, fontsize=11, pad=0)
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)


# =====================================================================
# 三、图 1：旋转到底对物体做了什么
# =====================================================================

def panel_effect(ax, w):
    R = rodrigues(w)
    k, theta = rotation_axis_angle(R)

    # 旋转前的盒子（灰色虚线）
    ax.add_collection3d(Line3DCollection(box_edges(), colors="#aaaaaa",
                                         linestyles="--", linewidths=1.0))
    # 旋转后的盒子（蓝色实线）
    ax.add_collection3d(Line3DCollection(box_edges(R), colors="#1f77b4",
                                         linewidths=1.7))

    # 坐标轴：细的 = 原坐标轴，粗的 = 转过去的坐标轴
    draw_axes(ax, None, length=1.30, lw=1.1, alpha=0.40)
    draw_axes(ax, R, length=1.30, lw=2.2, label=True)

    # 转轴：穿过原点的红色点线
    ax.plot([0, 1.55 * k[0]], [0, 1.55 * k[1]], [0, 1.55 * k[2]],
            color="#d62728", linestyle=":", linewidth=2)

    # 取盒子的一个角点，画它扫过的圆弧
    p0 = CORNERS[np.argmax(np.linalg.norm(CORNERS, axis=1))]   # 离原点最远的角
    ts = np.linspace(0, 1, 60)
    arc = np.array([rodrigues(t * w) @ p0 for t in ts])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="#ff7f0e", linewidth=2)
    ax.scatter(*p0, color="#ff7f0e", s=25)
    ax.scatter(*(R @ p0), color="#ff7f0e", s=25)

    # 文字说明放在轴外的固定位置，避免压住物体
    ax.text2D(0.02, 0.93, "红点线 = 转轴 k", transform=ax.transAxes,
              fontsize=9, color="#d62728")
    ax.text2D(0.02, 0.86, "橙弧 = 角点 p 的轨迹", transform=ax.transAxes,
              fontsize=9, color="#ff7f0e")

    style_3d(ax, 1.65,
             f"图 1  旋转的作用：绕 k 转 {np.degrees(theta):.0f}°\n"
             f"灰虚线 = 转之前，蓝实线 = 转之后", zoom=1.5)

    # 打印一些数字，方便和笔记对照
    print(f"[图1] w = {w.round(3)}  theta = {np.degrees(theta):.1f} deg")
    print(f"      k = {k.round(3)}")
    print(f"      det(R) = {np.linalg.det(R):.6f}   "
          f"R^T R = I ? {np.allclose(R.T @ R, np.eye(3))}")


# =====================================================================
# 四、图 2：绕轴连续转动，向量扫出一段圆弧
# =====================================================================

def panel_sequence(ax, w):
    """
    取一根从原点出发的向量 p，让它绕转轴 k 每次多转 15°。
    所有时刻的 p 连起来，就是它在空间里划过的圆弧 —— 这就是"绕轴旋转"最直观的样子。
    """
    axis = w / np.linalg.norm(w)
    p = np.array([1.25, 0.25, 0.0])

    angles = np.deg2rad(np.arange(0, 136, 15))
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(angles)))

    # 先画圆弧，再画箭头，箭头才不会被线挡住
    tips = np.array([rodrigues(axis * a) @ p for a in angles])
    ax.plot(tips[:, 0], tips[:, 1], tips[:, 2],
            color="#444444", linewidth=1.2, alpha=0.7)

    for a, c in zip(angles, colors):
        v = rodrigues(axis * a) @ p
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=[c],
                  linewidth=1.6, arrow_length_ratio=0.10)

    # 转轴
    ax.plot([-1.6 * axis[0], 1.6 * axis[0]],
            [-1.6 * axis[1], 1.6 * axis[1]],
            [-1.6 * axis[2], 1.6 * axis[2]],
            color="#d62728", linestyle=":", linewidth=2)

    # 用散点伪造图例
    for a, c in zip(angles[::3], colors[::3]):
        ax.scatter([], [], color=[c], label=f"{np.rad2deg(a):.0f}°")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2,
              handletextpad=0.2, columnspacing=0.8)

    style_3d(ax, 1.65, "图 2  绕轴连续转动\n同一个向量每 15° 扫一次，端点连成圆弧",
             zoom=1.5)


# =====================================================================
# 五、图 3：SO(3) 的"球"——旋转向量空间
# =====================================================================

def panel_ball(ax):
    # 半径 π 的球面（网格线）
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = np.pi * np.outer(np.cos(u), np.sin(v))
    y = np.pi * np.outer(np.sin(u), np.sin(v))
    z = np.pi * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#888888", linewidth=0.4, alpha=0.30)

    # z=0 平面上的"赤道"，帮助看方向
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.pi * np.cos(t), np.pi * np.sin(t), 0 * t,
            color="#bbbbbb", linewidth=0.8, alpha=0.8)

    # 几个"旋转向量"点（名字用 ω1/ω2，避免下标字符缺字体）
    pts = [
        (np.array([0.9, 0.5, 0.4]), " w1"),
        (np.array([-0.7, 0.2, 0.6]), " w2"),
        (np.array([0.0, 0.0, np.pi]), " pi*z"),
        (np.array([0.0, 0.0, -np.pi]), " -pi*z"),
    ]
    for p, name in pts:
        ax.quiver(0, 0, 0, p[0], p[1], p[2], color="#1f77b4",
                  linewidth=1.8, arrow_length_ratio=0.08)
        ax.text(p[0] * 1.10, p[1] * 1.10, p[2] * 1.10, name,
                fontsize=9, color="#1f77b4")

    # 对径点：同一个旋转
    ax.plot([0, 0], [0, 0], [np.pi, -np.pi], color="#d62728",
            linestyle="--", linewidth=1.4)

    # 从球心到 w1 的直线 = SO(3) 上的测地线
    p = pts[0][0]
    ax.plot([0, p[0]], [0, p[1]], [0, p[2]],
            color="#2ca02c", linestyle="-.", linewidth=1.6)

    # 说明文字统一放在轴外的 2D 位置，避免和球/箭头打架
    ax.text2D(0.02, 0.97, "球内：一个点 = 一个旋转", transform=ax.transAxes,
              fontsize=9, color="#333333")
    ax.text2D(0.02, 0.90, "球面：w 与 -w 是同一个旋转", transform=ax.transAxes,
              fontsize=9, color="#d62728")
    ax.text2D(0.02, 0.83, "球心：不转（θ=0）", transform=ax.transAxes,
              fontsize=9, color="#333333")
    ax.text2D(0.55, 0.10, "绿色点划线 = 测地线\n它的长度就是转角",
              transform=ax.transAxes, fontsize=9, color="#2ca02c")

    style_3d(ax, np.pi * 1.0, "图 3  SO(3) 的“球”：‖ω‖ ≤ π", zoom=1.22)


# =====================================================================
# 六、图 4：为什么小角度可以用一阶近似
# =====================================================================

def panel_linearization(ax):
    thetas = np.linspace(0, np.pi, 300)
    k = np.array([0.0, 0.0, 1.0])          # 绕 z 轴，方便看
    K = hat(k)

    err1, err2 = [], []
    for th in thetas:
        R = rodrigues(th * k)
        approx1 = np.eye(3) + th * K                                # 一阶
        approx2 = np.eye(3) + th * K + 0.5 * (th ** 2) * (K @ K)    # 二阶
        err1.append(np.linalg.norm(R - approx1))                    # Frobenius 范数
        err2.append(np.linalg.norm(R - approx2))

    ax.plot(np.degrees(thetas), err1, label="一阶：I + [ω]×", color="#d62728")
    ax.plot(np.degrees(thetas), err2, label="二阶：再加 ½[ω]×²", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("转过的角度（度）")
    ax.set_ylabel("‖R − 近似‖（对数）")
    ax.set_title("图 4  小角度线性化的误差\n10° 以内一阶近似就够用", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # 标出 10° 的位置
    i = int(np.argmin(np.abs(np.degrees(thetas) - 10)))
    ax.axvline(10, color="#999999", linestyle=":")
    ax.annotate(f"10°：{err1[i]:.1e}", xy=(10, err1[i]),
                xytext=(32, err1[i] * 4), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#666666"))
    print(f"[图4] 10° 时一阶近似误差 = {err1[i]:.3e}")
    print(f"      90° 时一阶近似误差 = "
          f"{err1[int(np.argmin(np.abs(np.degrees(thetas) - 90)))]:.3e}")


# =====================================================================
# 七、主流程
# =====================================================================

def main():
    fig = plt.figure(figsize=(13.5, 13))
    fig.suptitle("SO(3) 长什么样：旋转作用在 3D 空间 + 旋转向量空间",
                 fontsize=15, y=0.975)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4)

    # 用一个"斜着"的旋转，三个轴都参与，看起来最直观
    w = np.array([0.6, -0.8, 1.1])          # 轴约 (0.40,-0.54,0.74)，角度约 85°
    panel_effect(ax1, w)
    panel_sequence(ax2, w)
    panel_ball(ax3)
    panel_linearization(ax4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.895, bottom=0.07,
                        wspace=0.02, hspace=0.38)

    if SAVE:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "so3_visual.png")
        fig.savefig(out, dpi=150)
        print(f"\n已保存: {out}")

    # 默认直接显示（关掉窗口后程序才结束）；--save 时不弹窗
    if not SAVE:
        plt.show()
    print("ALL DONE")


if __name__ == "__main__":
    main()
