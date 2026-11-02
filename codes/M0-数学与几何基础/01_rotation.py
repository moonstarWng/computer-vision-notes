# -*- coding: utf-8 -*-
"""
M0 第 1 篇 · 旋转：表示、互转、优化
对应笔记：notes/M0-数学与几何基础/1-旋转.md（§2 六种表示、§6 代码实验）

这个脚本回答三个问题
--------------------
1. 旋转向量 ω = θk 怎么变成旋转矩阵 R？反过来怎么取回 ω？
2. 旋转向量、四元数、旋转矩阵互换，会不会丢信息？
3. 为什么"det = 1"不能证明一个矩阵是旋转？

运行
----
    python 01_rotation.py

预期：最后一行打印 ALL PASS；中途任何断言失败会直接报错并给出原因。

阅读顺序
--------
hat  ->  rodrigues  ->  log_so3  ->  quat_from_rotvec / rotvec_from_quat
（函数定义在前，演示在后面的 demo_* 里，可以只看 main 的调用顺序）
"""

import numpy as np


# =====================================================================
# 一、基础算子：向量和矩阵之间的"翻译器"
# =====================================================================

def hat(w):
    """
    三维向量 -> 反对称矩阵（也叫叉乘矩阵）

        数学定义：
                [  0   -w3   w2 ]
        [w]x =  [ w3    0   -w1 ]
                [-w2   w1    0  ]

    为什么要这个东西？
        矩阵只能做乘法，不能做叉乘。把 w 换成 [w]x 之后：
            [w]x @ v  ==  w × v
        于是"叉乘"就变成了"矩阵乘法"，后面才能对旋转写导数（雅可比）。

    参数：w，长度为 3 的向量
    返回：3×3 反对称矩阵（它转置一下就是自己的相反数：W.T == -W）
    """
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(W):
    """
    反对称矩阵 -> 三维向量（hat 的逆运算）

    只挑出 (2,1)、(0,2)、(1,0) 三个位置：
        W[2,1] = w1,  W[0,2] = w2,  W[1,0] = w3
    这三个数正好是原向量。顺序写错会得到 -w，这是很常见的 bug。

    参数：3×3 反对称矩阵
    返回：长度为 3 的向量
    """
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


# =====================================================================
# 二、指数映射 / 对数映射（旋转向量 <-> 旋转矩阵）
# =====================================================================

def rodrigues(w):
    """
    旋转向量 ω = θk  ->  旋转矩阵 R，即 exp(ω^)

    罗德里格斯公式：
        R = I + sinθ·[k]x + (1 − cosθ)·[k]x²

    参数含义：
        θ = ‖ω‖        转角（弧度）
        k = ω / ‖ω‖    单位旋转轴

    两个边界要小心：
        1) θ = 0：k = ω/θ 会除零 —— 任何轴都可以，结果就是单位阵 I；
        2) 结果一定是合法旋转（det=1 且 R^T R=I），不用额外修正。

    参数：w，长度为 3 的旋转向量
    返回：3×3 旋转矩阵
    """
    theta = np.linalg.norm(w)          # θ = ‖ω‖，向量的长度就是转角
    if theta < 1e-12:                  # 跳过除零：ω≈0 时 exp(0)=I
        return np.eye(3)

    k = w / theta                      # 单位轴
    K = hat(k)                         # [k]x
    # 公式逐项抄下来：I + sinθ·K + (1−cosθ)·K²
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R):
    """
    旋转矩阵 R  ->  旋转向量 ω，即 log(R)

    三个公式：
        θ = arccos( (tr(R) − 1) / 2 )              转角
        ω = θ / (2·sinθ) · (R − R^T)∨              轴 × 角

    为什么长这样：对上式求逆，利用 R − R^T = 2·sinθ·[k]x。

    三个边界：
        θ = 0   ：R≈I，直接返回 0 向量（此时轴没有意义）；
        θ = π   ：sinθ≈0，分母又炸了。此时 R + I 的每一列都正比于 k，
                  所以取"长度最大的那一列"归一化，就能拿到轴；
        θ 在中间：直接用上面的公式。

    返回值一定满足 ‖ω‖ ≤ π（这一点在优化里很重要，否则变量会"漂"）。

    参数：3×3 旋转矩阵
    返回：长度为 3 的旋转向量
    """
    # trace = 对角线之和；用 clip 防止数值误差让 cos 落到 [-1,1] 外面
    cos_theta = (np.trace(R) - 1.0) / 2.0
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    if theta < 1e-12:                  # 情况 1：θ≈0
        return np.zeros(3)

    if np.pi - theta < 1e-6:           # 情况 2：θ≈π，换一条公式
        A = R + np.eye(3)              # A 的列都平行于轴 k
        norms = np.linalg.norm(A, axis=0)   # axis=0 表示"按列"算长度
        k = A[:, np.argmax(norms)]     # 取最长的那一列，最不容易被噪声污染
        return np.pi * (k / np.linalg.norm(k))

    # 情况 3：一般情形。vee(R − R^T) = 2·sinθ·k
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


# =====================================================================
# 三、四元数（存储/插值用）
# =====================================================================

def quat_from_rotvec(w):
    """
    旋转向量 -> 单位四元数，约定顺序为 (w, x, y, z)，即"标量在前"

    公式：
        q = ( cos(θ/2), sin(θ/2)·k )

    最容易错的地方：角度是 θ/2，不是 θ。因为四元数作用时是 q v q*，
    转两次才等于转 θ（180° 的旋转对应 360° 的四元数）。

    参数：w，长度为 3 的旋转向量
    返回：长度为 4 的数组 [w, x, y, z]，模长为 1
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:                  # θ≈0 -> 单位四元数 (1,0,0,0)
        return np.array([1.0, 0.0, 0.0, 0.0])
    k = w / theta
    # np.concatenate 把"标量"和"3 维向量"拼成一个 4 维数组
    return np.concatenate([[np.cos(theta / 2.0)], np.sin(theta / 2.0) * k])


def rotvec_from_quat(q):
    """
    单位四元数 -> 旋转向量

    做法：
        1) 先统一符号：q 和 −q 表示同一个旋转，但算出来的角度可能是
           2π−θ（绕远路）。约定统一取标量部分 w ≥ 0 的那一个；
        2) 由 w = cos(θ/2) 得 θ = 2·arccos(w)；
        3) 轴就是向量部分 v 归一化后的方向。

    这里用 arctan2(s, w) 而不是 arccos(w)，是为了在 w 接近 1 时更稳定。

    参数：q，(w, x, y, z) 顺序
    返回：长度为 3 的旋转向量，‖ω‖ ≤ π
    """
    q = np.asarray(q, dtype=float)
    if q[0] < 0:                       # 符号对齐：q 与 −q 是同一个旋转
        q = -q

    v = q[1:]                          # 向量部分 (x, y, z)
    s = np.linalg.norm(v)              # s = sin(θ/2)
    if s < 1e-12:                      # θ≈0
        return np.zeros(3)

    theta = 2.0 * np.arctan2(s, q[0])  # θ = 2·atan2(sin(θ/2), cos(θ/2))
    return theta * v / s               # θ · 单位轴


def check_rotation(R, name="R", atol=1e-6):
    """
    检查一个 3×3 矩阵到底是不是"旋转"（这是工程防线，不是数学练习）

    两个条件缺一不可：
        RᵀR = I    正交、不拉伸、不剪切
        det(R) = +1 不镜像（det=-1 是反射，物理上做不到）

    参数：R 待检查矩阵；name 报错时用来指明是谁
    返回：None（不合格会抛 AssertionError）
    """
    assert np.allclose(R.T @ R, np.eye(3), atol=atol), \
        f"{name} 不正交（有剪切/拉伸）"
    d = np.linalg.det(R)
    assert d > 0 and abs(d - 1.0) < atol, f"{name} 是镜像！det={d:.6f}"


# =====================================================================
# 四、演示：每个 demo 对应笔记里的一个结论
# =====================================================================

def demo_basic():
    """绕 z 轴转 90°：手算能对上的最基本例子"""
    print("== 1. rodrigues：绕 z 转 90° ==")
    R = rodrigues(np.array([0.0, 0.0, np.pi / 2]))   # 轴是 z，角度 π/2
    print(R.round(3))                                # 应该看到 [[0,-1,0],[1,0,0],[0,0,1]]
    # det 用来看"有没有镜像"，正交用来看"有没有剪切"
    print(f"det = {np.linalg.det(R):.6f}  正交 = {np.allclose(R.T @ R, np.eye(3))}")
    # log 应该把 π/2 这个角度原样还回来
    print(f"log 回去 = {log_so3(R).round(4)}\n")


def demo_roundtrip(seed=0):
    """
    随机抽 2000 个旋转，走 R -> ω -> R 一圈，看最大误差。
    误差应该在 1e-11 量级（只有浮点舍入），说明 exp/log 互逆。
    """
    print("== 2. exp / log 往返 2000 次 ==")
    rng = np.random.default_rng(seed)          # 固定种子，结果可复现
    err = 0.0
    for _ in range(2000):
        axis = rng.normal(size=3)              # 随机方向
        axis /= np.linalg.norm(axis)           # 归一化成单位轴
        w = axis * rng.uniform(0.0, np.pi * 0.999)   # 随机角度，避免正好 π
        err = max(err, np.linalg.norm(rodrigues(log_so3(rodrigues(w))) - rodrigues(w)))
    print(f"max |exp(log(R)) - R| = {err:.2e}\n")


def demo_quaternion(seed=1):
    """旋转向量 -> 四元数 -> 旋转向量，验证不丢信息"""
    print("== 3. 旋转向量 <-> 四元数 ==")
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(1000):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * rng.uniform(0.0, np.pi * 0.99)
        err = max(err, np.linalg.norm(rotvec_from_quat(quat_from_rotvec(w)) - w))
    print(f"max |w - q2rotvec(rotvec2q(w))| = {err:.2e}\n")


def demo_pi_ambiguity():
    """
    θ = π 时，ω 和 −ω 是同一个旋转（球面上对径点粘合）。
    所以 log 在 π 附近不唯一，这是"旋转向量不能当全局唯一表示"的根本原因。
    """
    print("== 4. θ=π 的歧义：ω 与 -ω 是同一个旋转 ==")
    Ra = rodrigues(np.array([0.0, 0.0, np.pi]))
    Rb = rodrigues(np.array([0.0, 0.0, -np.pi]))
    print(f"exp(πz) == exp(-πz): {np.allclose(Ra, Rb)}\n")


def demo_det_trap():
    """
    在 R 上直接加一个"只有 (1,1) 位置为 1"的扰动：
    行列式居然还是 1，但 RᵀR 已经不是单位阵了 —— 说明它被"剪切"了。
    结论：det=1 是必要条件，不是充分条件。
    """
    print("== 5. det=1 不代表是旋转（剪切反例） ==")
    R = rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    E11 = np.diag([1.0, 0.0, 0.0])             # 3×3 对角阵，只有 [0,0] 是 1
    R2 = R + 0.1 * E11                         # 只扰动一个元素
    print(f"det(R2) = {np.linalg.det(R2):.4f}  <- 还是 1")
    print(f"R2^T R2 正交 = {np.allclose(R2.T @ R2, np.eye(3))}  <- 但已经不是单位阵")
    print((R2.T @ R2).round(3), "\n")          # 非对角元的 -0.1 就是"剪切"


def demo_scipy_chain():
    """和工业库 scipy 对一下：四种表示互相转换的结果是否一致"""
    print("== 6. scipy 走一圈（R / q / omega / 欧拉角） ==")
    try:
        from scipy.spatial.transform import Rotation as Rot
    except ImportError:
        # scipy 只是用来对照的，没装也不影响前面几节
        print("未安装 scipy，已跳过（pip install scipy）\n")
        return

    R = Rot.from_rotvec([0, 0, np.pi / 2]).as_matrix()     # 旋转向量 -> 矩阵
    q = Rot.from_matrix(R).as_quat()                       # 注意：scipy 是 xyzw 顺序
    e = Rot.from_matrix(R).as_euler("zyx", degrees=True)   # yaw, pitch, roll
    w = Rot.from_matrix(R).as_rotvec()                     # 矩阵 -> 旋转向量
    print(f"q(xyzw) = {q.round(4)}")
    print(f"euler   = {e.round(1)}")
    print(f"rotvec  = {w.round(4)}")
    # 再用四元数转回矩阵，应该和最初的 R 完全一致
    print(f"回代一致 = {np.allclose(Rot.from_quat(q).as_matrix(), R)}\n")


def main():
    demo_basic()
    demo_roundtrip()
    demo_quaternion()
    demo_pi_ambiguity()
    demo_det_trap()
    demo_scipy_chain()

    # 最后交叉验证：旋转向量路径 和 四元数路径 算出的 R 必须相同
    w = np.array([0.3, -0.5, 1.1])
    R1 = rodrigues(w)
    q = quat_from_rotvec(w)
    R2 = rodrigues(rotvec_from_quat(q))
    check_rotation(R1)          # 断言：R1 是合法旋转
    check_rotation(R2)          # 断言：R2 也是
    assert np.allclose(R1, R2, atol=1e-12), "旋转向量与四元数两条路不一致"
    print("ALL PASS")


if __name__ == "__main__":
    main()
