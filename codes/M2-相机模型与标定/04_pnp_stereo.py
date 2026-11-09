# -*- coding: utf-8 -*-
"""
M2 第 4 篇 —— 外参、PnP 与双目

对应笔记：notes/M2-相机模型与标定/4-外参-PnP与双目.md

这个脚本回答四个问题
--------------------
1. 已知 3D 点和像素，怎么解外参（R, t）？DLT 和迭代法各干什么？
2. 有外点时怎么办？RANSAC 为什么能把它们挑出来？
3. 双目的 Z = fb/d 怎么来的？为什么远处精度会崩？
4. 同一个误差 0.1 px，在 1 m 和 5 m 处差多少？

运行：
    python 04_pnp_stereo.py
预期：最后打印 ALL PASS。
只依赖 numpy。
"""

import numpy as np

RNG = np.random.default_rng(11)

K = np.array([[2318.84, 0.0, 639.5],
              [0.0, 2318.84, 511.5],
              [0.0, 0.0, 1.0]])
F_MM, BASELINE_MM = 8.0, 60.0


# =====================================================================
# 一、基础
# =====================================================================

def rodrigues(w):
    """【rodrigues】axis-angle -> R"""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def log_so3(R):
    """【log_so3】R -> axis-angle"""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * np.sin(th)) * th


def project(K, rvec, tvec, X):
    """【project】世界点 -> 像素"""
    Xc = np.asarray(X, float) @ rodrigues(rvec).T + np.asarray(tvec, float)
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def rot_error_deg(Ra, Rb):
    """【rot_error_deg】两个旋转之间的夹角（度）"""
    R = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# =====================================================================
# 二、PnP：DLT（已知 K）+ LM 精化
# =====================================================================

def pnp_dlt(K, X, uv):
    """
    【pnp_dlt】PnP DLT → 用已知内参，从 3D-2D 对应解 [R|t]

    思路：先用 K⁻¹ 把像素变成归一化坐标，问题就退化成"已知内参的 PnP"：

        x_n = (m1·X)/(m3·X),  y_n = (m2·X)/(m3·X)

    把 m 的 12 个元素当未知数，每个点给 2 个齐次方程，≥6 点即可。
    """
    X = np.asarray(X, float)
    Kinv = np.linalg.inv(K)
    h = np.hstack([uv, np.ones((len(uv), 1))]) @ Kinv.T
    xn = h[:, :2] / h[:, 2:3]

    A = []
    for (Xi, x, y) in zip(X, xn[:, 0], xn[:, 1]):
        X4 = np.append(Xi, 1.0)
        A.append(np.concatenate([X4, np.zeros(4), -x * X4]))
        A.append(np.concatenate([np.zeros(4), X4, -y * X4]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    M = Vt[-1].reshape(3, 4)

    # M = s·[R|t]，但 P 与 -P 给出完全相同的投影：先选使 det>0 的那个符号。
    # 注意：不能只对 R 做 SO(3) 投影了事——那会把 R 改掉而 t 没改，姿态会翻 180°。
    if np.linalg.det(M[:, :3]) < 0:
        M = -M
    s = np.linalg.norm(M[:, :3], axis=0).mean()
    R = M[:, :3] / s
    t = M[:, 3] / s
    U, _, Vt2 = np.linalg.svd(R)
    R = U @ Vt2                      # 轻微正交化（噪声导致的小偏差）
    return R, t


def pnp_refine(K, X, uv, R0, t0, iters=30):
    """【pnp_refine】LM 最小化重投影误差，精化 (R, t)"""
    p = np.concatenate([log_so3(R0), t0])
    lam = 1e-3

    def resid(q):
        return (project(K, q[:3], q[3:], X) - uv).ravel()

    cost = np.sum(resid(p) ** 2)
    for _ in range(iters):
        r = resid(p)
        J = np.zeros((len(r), 6))
        for j in range(6):
            dp = np.zeros(6)
            dp[j] = 1e-6 * max(1.0, abs(p[j]))
            J[:, j] = (resid(p + dp) - r) / dp[j]
        A = J.T @ J
        g = J.T @ r
        ok = False
        for _ in range(8):
            try:
                d = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            if np.sum(resid(p + d) ** 2) < cost:
                p = p + d
                cost = np.sum(resid(p) ** 2)
                lam = max(lam / 3, 1e-12)
                ok = True
                break
            lam *= 5
        if not ok:
            break
    return rodrigues(p[:3]), p[3:], np.sqrt(cost / (2 * len(X)))


# =====================================================================
# 三、RANSAC
# =====================================================================

def ransac_pnp(K, X, uv, thresh_px=3.0, iters=300, min_sample=6, seed=0):
    """
    【ransac_pnp】RANSAC + PnP → 有外点时也能解

    每次随机抽 6 个点解 PnP，数"重投影误差 < 阈值"的内点。
    内点最多的一组当最终解，再用全部内点精化。
    """
    r = np.random.default_rng(seed)
    best_inl = None
    best_cnt = -1
    for _ in range(iters):
        idx = r.choice(len(X), min_sample, replace=False)
        try:
            R, t = pnp_dlt(K, X[idx], uv[idx])
        except np.linalg.LinAlgError:
            continue
        e = np.linalg.norm(project(K, log_so3(R), t, X) - uv, axis=1)
        inl = e < thresh_px
        if inl.sum() > best_cnt:
            best_cnt = int(inl.sum())
            best_inl = inl
    R, t, _ = pnp_refine(K, X[best_inl], uv[best_inl], *pnp_dlt(K, X[best_inl], uv[best_inl]))

    # 用精化后的模型重算一遍内点，再精化一次（标准做法：内点集也要迭代）
    e = np.linalg.norm(project(K, log_so3(R), t, X) - uv, axis=1)
    inl = e < thresh_px
    R, t, _ = pnp_refine(K, X[inl], uv[inl], R, t)
    return R, t, inl


# =====================================================================
# 四、demo
# =====================================================================

def make_scene(n=20):
    """【make_scene】在 200mm 立方体内撒点，并给定真实位姿"""
    r = np.random.default_rng(3)
    X = r.uniform(-100, 100, (n, 3))
    X[:, 2] += 300.0                 # 物体在相机前方 200~400 mm
    rvec_true = np.array([0.10, -0.20, 0.30])
    tvec_true = np.array([20.0, -30.0, 600.0])
    return X, rvec_true, tvec_true


def demo_pnp():
    print("一、PnP：DLT 与精化")
    X, rvec_true, tvec_true = make_scene(20)
    uv = project(K, rvec_true, tvec_true, X) + RNG.normal(0, 0.5, (20, 2))
    print("  点数 {}，角点噪声 0.5 px".format(len(X)))
    print("  真值: |rvec|={:.1f}°  t=({:.1f}, {:.1f}, {:.1f}) mm".format(
        np.degrees(np.linalg.norm(rvec_true)), *tvec_true))

    R0, t0 = pnp_dlt(K, X, uv)
    e0 = np.linalg.norm(project(K, log_so3(R0), t0, X) - uv, axis=1).mean()
    print("  DLT  : 旋转误差 {:.3f}°，平移误差 {:.2f} mm，平均重投影 {:.3f} px".format(
        rot_error_deg(R0, rodrigues(rvec_true)), np.linalg.norm(t0 - tvec_true), e0))

    R1, t1, rmse = pnp_refine(K, X, uv, R0, t0)
    print("  精化 : 旋转误差 {:.3f}°，平移误差 {:.2f} mm，RMSE {:.3f} px".format(
        rot_error_deg(R1, rodrigues(rvec_true)), np.linalg.norm(t1 - tvec_true), rmse))
    assert rmse < 0.6
    return X, rvec_true, tvec_true, uv


def demo_ransac():
    print("\n二、有外点时：DLT 报废，RANSAC 能救")
    X, rvec_true, tvec_true = make_scene(30)
    uv = project(K, rvec_true, tvec_true, X) + RNG.normal(0, 0.5, (30, 2))

    # 30% 外点：随机撒在图像里
    n_out = 9
    idx = RNG.choice(len(X), n_out, replace=False)
    uv_bad = uv.copy()
    uv_bad[idx] = np.column_stack([RNG.uniform(0, 1280, n_out), RNG.uniform(0, 1024, n_out)])
    print("  总点数 {}，外点 {}（{:.0%}）".format(len(X), n_out, n_out / len(X)))

    R0, t0 = pnp_dlt(K, X, uv_bad)
    print("  直接 DLT: 旋转误差 {:8.1f}°，平移误差 {:8.1f} mm".format(
        rot_error_deg(R0, rodrigues(rvec_true)), np.linalg.norm(t0 - tvec_true)))

    R1, t1, inl = ransac_pnp(K, X, uv_bad, thresh_px=3.0, iters=400, seed=5)
    print("  RANSAC  : 旋转误差 {:8.3f}°，平移误差 {:8.2f} mm，内点 {}/{}".format(
        rot_error_deg(R1, rodrigues(rvec_true)), np.linalg.norm(t1 - tvec_true),
        int(inl.sum()), len(X)))
    assert inl.sum() >= len(X) - n_out
    assert rot_error_deg(R1, rodrigues(rvec_true)) < 1.0


def demo_stereo():
    print("\n三、双目：Z = fx·b / d 的精度")
    fx = K[0, 0]
    print("  焦距 f={:.1f} mm，基线 b={:.0f} mm，fx={:.2f} px".format(F_MM, BASELINE_MM, fx))
    print("   Z(mm) | 视差 d(px) | Δd=0.1px 时 ΔZ | Δd=0.5px 时 ΔZ")
    for Z in (500.0, 1000.0, 2000.0, 5000.0):
        d = fx * BASELINE_MM / Z
        dz01 = Z * Z * 0.1 / (fx * BASELINE_MM)
        dz05 = Z * Z * 0.5 / (fx * BASELINE_MM)
        print("  {:6.0f} | {:9.2f}   | {:10.1f} mm  | {:10.1f} mm".format(Z, d, dz01, dz05))

    # 精度随 Z² 增长：2 倍距离 -> 4 倍误差
    z1, z2 = 1000.0, 2000.0
    e1 = z1 * z1 * 0.1 / (fx * BASELINE_MM)
    e2 = z2 * z2 * 0.1 / (fx * BASELINE_MM)
    print("  距离翻倍，误差 {:.1f} -> {:.1f} mm（{:.1f} 倍）".format(e1, e2, e2 / e1))
    assert np.isclose(e2 / e1, 4.0)


def main():
    demo_pnp()
    demo_ransac()
    demo_stereo()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
