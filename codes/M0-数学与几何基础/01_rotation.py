"""
M0 第 1 篇 · 旋转：表示、互转、优化
对应笔记：notes/M0-数学与几何基础/1-旋转.md（§2、§6）

运行：
    python 01_rotation.py

全部通过时最后一行打印 ALL PASS。
"""

import numpy as np


# ---------------------------------------------------------------- 基础算子
def hat(w):
    """向量 → 反对称矩阵：hat(w) @ v == cross(w, v)"""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(W):
    """反对称矩阵 → 向量（hat 的逆）"""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def rodrigues(w):
    """旋转向量 ω = θk → 旋转矩阵 R，也就是 exp(ω^)"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_so3(R):
    """旋转矩阵 R → 旋转向量 ω，也就是 log(R)，结果满足 ||ω|| <= π"""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:                 # θ≈π：取 R+I 的最大列做轴
        A = R + np.eye(3)
        k = A[:, np.argmax(np.linalg.norm(A, axis=0))]
        return np.pi * (k / np.linalg.norm(k))
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def quat_from_rotvec(w):
    """旋转向量 → 单位四元数 (w, x, y, z)，注意角度是 θ/2"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    k = w / theta
    return np.concatenate([[np.cos(theta / 2.0)], np.sin(theta / 2.0) * k])


def rotvec_from_quat(q):
    """单位四元数 → 旋转向量（先统一到 w>=0 的那一半）"""
    q = np.asarray(q, dtype=float)
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(s, q[0])
    return theta * v / s


def check_rotation(R, name="R", atol=1e-6):
    """任何外参进系统前都该过一遍"""
    assert np.allclose(R.T @ R, np.eye(3), atol=atol), f"{name} 不正交（有剪切/拉伸）"
    d = np.linalg.det(R)
    assert d > 0 and abs(d - 1.0) < atol, f"{name} 是镜像！det={d:.6f}"


# ---------------------------------------------------------------- 演示
def demo_basic():
    print("== 1. rodrigues：绕 z 转 90° ==")
    R = rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    print(R.round(3))
    print(f"det = {np.linalg.det(R):.6f}  正交 = {np.allclose(R.T @ R, np.eye(3))}")
    print(f"log 回去 = {log_so3(R).round(4)}\n")


def demo_roundtrip(seed=0):
    print("== 2. exp / log 往返 2000 次 ==")
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(2000):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = axis * rng.uniform(0.0, np.pi * 0.999)
        err = max(err, np.linalg.norm(rodrigues(log_so3(rodrigues(w))) - rodrigues(w)))
    print(f"max |exp(log(R)) - R| = {err:.2e}\n")


def demo_quaternion(seed=1):
    print("== 3. 旋转向量 <-> 四元数 ==")
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(1000):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * rng.uniform(0.0, np.pi * 0.99)
        err = max(err, np.linalg.norm(rotvec_from_quat(quat_from_rotvec(w)) - w))
    print(f"max |ω - q2rotvec(rotvec2q(ω))| = {err:.2e}\n")


def demo_pi_ambiguity():
    print("== 4. θ=π 的歧义：ω 与 -ω 是同一个旋转 ==")
    Ra = rodrigues(np.array([0.0, 0.0, np.pi]))
    Rb = rodrigues(np.array([0.0, 0.0, -np.pi]))
    print(f"exp(πz) == exp(-πz): {np.allclose(Ra, Rb)}\n")


def demo_det_trap():
    print("== 5. det=1 不代表是旋转（剪切反例） ==")
    R = rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    E11 = np.diag([1.0, 0.0, 0.0])
    R2 = R + 0.1 * E11
    print(f"det(R2) = {np.linalg.det(R2):.4f}  ← 还是 1")
    print(f"R2^TR2 正交 = {np.allclose(R2.T @ R2, np.eye(3))}  ← 但已经不是单位阵")
    print((R2.T @ R2).round(3), "\n")


def demo_scipy_chain():
    print("== 6. scipy 走一圈（R / q / ω / 欧拉角） ==")
    try:
        from scipy.spatial.transform import Rotation as Rot
    except ImportError:
        print("未安装 scipy，已跳过（pip install scipy）\n")
        return
    R = Rot.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    q = Rot.from_matrix(R).as_quat()                      # xyzw 顺序
    e = Rot.from_matrix(R).as_euler("zyx", degrees=True)  # yaw, pitch, roll
    w = Rot.from_matrix(R).as_rotvec()
    print(f"q(xyzw) = {q.round(4)}")
    print(f"euler   = {e.round(1)}")
    print(f"rotvec  = {w.round(4)}")
    print(f"回代一致 = {np.allclose(Rot.from_quat(q).as_matrix(), R)}\n")


def main():
    demo_basic()
    demo_roundtrip()
    demo_quaternion()
    demo_pi_ambiguity()
    demo_det_trap()
    demo_scipy_chain()

    # 一致性检查：三条路算出的 R 必须相同
    w = np.array([0.3, -0.5, 1.1])
    R1 = rodrigues(w)
    q = quat_from_rotvec(w)
    R2 = rodrigues(rotvec_from_quat(q))
    check_rotation(R1)
    check_rotation(R2)
    assert np.allclose(R1, R2, atol=1e-12), "旋转向量与四元数两条路不一致"
    print("ALL PASS")


if __name__ == "__main__":
    main()
