# -*- coding: utf-8 -*-
"""
M2 第 5 篇 —— 手眼标定实战：单线激光 + 切割圆

对应笔记：notes/M2-相机模型与标定/5-手眼标定-切割圆实战.md

这个脚本回答六个问题
--------------------
1. 线激光只给 (X, 0, Z)，为什么还能把 6 自由度手眼标定解出来？
2. 它和 M2 第 4 篇的 AX = XB 是什么关系？为什么这里"纯平移不退化"？
3. 为什么现场的流程是"切 4 个圆、两个高度各扫一遍"？单高度行不行？
4. 什么时候会解不出来（残差很小但解是错的）？
5. H 定义转置了会发生什么（镜像解长什么样）？
6. 测量噪声 0.01 -> 0.2 mm，误差怎么涨？

运行：
    python 05_handeye_cutcircle.py
预期：最后打印 ALL PASS。
只依赖 numpy。
"""

import numpy as np

RNG = np.random.default_rng(42)

# =====================================================================
# 一、基础工具：旋转 / 变换 / Kabsch
# =====================================================================


def rodrigues(w):
    """【rodrigues】轴角 -> 旋转矩阵（单位：rad）"""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def log_so3(R):
    """【log_so3】旋转矩阵 -> 轴角（单位：rad）"""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * np.sin(th)) * th


def rot_error_deg(R1, R2):
    """【rot_error_deg】两个旋转之间的夹角（单位：度）"""
    return np.degrees(np.linalg.norm(log_so3(R1 @ R2.T)))


def make_T(R, t):
    """【make_T】R, t -> 4x4 齐次变换"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float)
    return T


def inv_T(T):
    """【inv_T】刚体变换的逆：R^T 与 -R^T t"""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def kabsch(p, q):
    """
    【kabsch】对应点已知时，闭式求 R, t 使 q ≈ R p + t

    p, q : (N, 3)，第 i 行是同一个物理点
    返回  : R(3x3), t(3,), 奇异值 S(3,), det 修正符号 d

    注意 H 的定义与 R 的公式必须配套：
        H = Σ p̃ q̃ᵀ  ->  R = V diag(1,1,det(VUᵀ)) Uᵀ
    换成 H = Σ q̃ p̃ᵀ 而不改公式，解出来是 Rᵀ（见 demo_transpose）。
    """
    p_bar, q_bar = p.mean(0), q.mean(0)
    p_t, q_t = p - p_bar, q - q_bar
    H = p_t.T @ q_t                      # 3x3 = Σ p̃ q̃ᵀ
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = float(np.sign(np.linalg.det(V @ U.T)))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_bar - R @ p_bar
    return R, t, S, d


def solve_handeye(p_cam, P_base, A_poses):
    """
    【solve_handeye】本项目的手眼求解：q_i = A_i⁻¹ P_i，再 Kabsch

    p_cam   : (N,3) 相机坐标下的圆心
    P_base  : (N,3) 基坐标下的圆心真值（切割给的）
    A_poses : (N,4,4) 每次扫描的机器人位姿 T_(B<-E)
    """
    q = np.array([(inv_T(A) @ np.append(P, 1.0))[:3] for P, A in zip(P_base, A_poses)])
    R, t, S, d = kabsch(p_cam, q)
    X = make_T(R, t)
    # 残差：把相机点变换到基坐标，再和真值比
    pred = np.array([(A @ X @ np.append(p, 1.0))[:3] for p, A in zip(p_cam, A_poses)])
    resid = np.linalg.norm(pred - P_base, axis=1)
    return X, resid, S, d


# =====================================================================
# 二、仿真数据：4 个圆 × 2 个高度 = 8 组对应
# =====================================================================

# 真值：安装面内 0.4°、面外 0.15° / 0.10°，平移 (0, 74, 170) mm
X_TRUE = make_T(rodrigues(np.radians([0.10, 0.15, 0.40])), [0.0, 74.0, 170.0])

# 4 个切割圆心（基坐标真值），排成一个 160 x 300 的矩形，都在管材上表面
SURFACE_Z = 120.0
CIRCLES_XY = [(80.0, -150.0), (-80.0, -50.0), (80.0, 50.0), (-80.0, 150.0)]
Z_LEVELS = (120.0, 160.0)          # 两次扫描时切割头停在的高度（现场 Z2 / Z1）


def laser_pose_for(P, z_tcp):
    """
    【laser_pose_for】给一个圆，求机器人应该停在哪个 Y，使激光面正好过圆心

    传感器平面是 XZ（y=0），法向 n = R_true · e_y。
    机器人姿态取单位阵，只平移：n·(P - t_A - t_true) = 0 解 t_A.y。
    这一步就是现场"上位机自动规划扫描位置"在做的事。
    """
    R, t = X_TRUE[:3, :3], X_TRUE[:3, 3]
    n = R @ np.array([0.0, 1.0, 0.0])
    t_ax, t_az = 0.0, z_tcp
    # n·(P - t_A - t) = 0  解 t_A.y（注意 u_y = P_y - t_Ay - t_y，所以是加号）
    t_ay = P[1] - t[1] + (n[0] * (P[0] - t_ax - t[0]) + n[2] * (P[2] - t_az - t[2])) / n[1]
    return make_T(np.eye(3), [t_ax, t_ay, t_az])


def make_dataset(noise=0.02, levels=Z_LEVELS, circles=None, seed=0):
    """
    【make_dataset】正演一组标定数据

    noise   : 传感器 2D 噪声的高斯标准差（mm）
    levels  : 扫描高度列表，一个高度 = 一组姿态（纯平移）
    circles : [(x, y), ...] 圆心平面位置
    返回    : p_cam(N,3), P_base(N,3), A_poses(N,4,4)
    """
    rng = np.random.default_rng(seed)
    circles = CIRCLES_XY if circles is None else circles
    p_cam, P_base, A_poses = [], [], []
    for z in levels:
        for cx, cy in circles:
            # 圆孔长在管材表面（Z = SURFACE_Z），两次扫描只是切割头高度不同
            P = np.array([cx, cy, SURFACE_Z])
            A = laser_pose_for(P, z)
            # P 在相机系下的坐标：x = X⁻¹ A⁻¹ P，y 应为 0（激光面过圆心）
            p = (inv_T(X_TRUE) @ inv_T(A) @ np.append(P, 1.0))[:3]
            # 只污染平面内的两个坐标：线激光测不到 y
            p[0] += rng.normal(0.0, noise)
            p[2] += rng.normal(0.0, noise)
            p_cam.append(p)
            P_base.append(P)
            A_poses.append(A)
    return np.array(p_cam), np.array(P_base), np.array(A_poses)


def report(tag, p_cam, P_base, A_poses, X_true=X_TRUE):
    """【report】解一次并打印一行结果"""
    X, resid, S, _ = solve_handeye(p_cam, P_base, A_poses)
    r_err = rot_error_deg(X[:3, :3], X_true[:3, :3])
    t_err = np.linalg.norm(X[:3, 3] - X_true[:3, 3])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ratio = S[1] / S[0]
    print("  {:22s} 点={:2d}  RMS={:7.4f} mm  旋转误差={:7.4f}°  平移误差={:7.4f} mm  "
          "σ2/σ1={:6.4f}".format(tag, len(P_base), rms, r_err, t_err, ratio))
    return dict(X=X, rms=rms, r_err=r_err, t_err=t_err, ratio=ratio, resid=resid, S=S)


# =====================================================================
# 三、六个问题，逐个演示
# =====================================================================


def demo_basic():
    print("\n一、基本标定：4 个圆 × 2 个高度 = 8 组对应，噪声 σ=0.02 mm")
    p, P, A = make_dataset(noise=0.02, seed=1)
    res = report("8 组（双高度）", p, P, A)
    print("  解出的手眼矩阵 X：")
    for row in res["X"]:
        print("    [{:9.4f} {:9.4f} {:9.4f} {:9.4f}]".format(*row))
    print("  奇异值 σ(H) = [{:.2f} {:.2f} {:.2e}]".format(*res["S"]))
    print("  逐点残差 (mm) = {}".format(np.round(res["resid"], 4)))

    # 残差是"毫米级很小"，但 σ3 恒为 0 —— 它不是退化信号
    assert res["S"][2] < 1e-6 * res["S"][0]
    assert res["r_err"] < 0.02 and res["t_err"] < 0.05


def demo_heights():
    print("\n二、单高度为什么不行：共面孔在传感器里退化成一条直线")
    p1, P1, A1 = make_dataset(noise=0.0, levels=(120.0,), seed=2)
    r1 = report("单高度（4 组）", p1, P1, A1)
    p2, P2, A2 = make_dataset(noise=0.0, levels=Z_LEVELS, seed=2)
    r2 = report("双高度（8 组）", p2, P2, A2)

    # 同一个高度，4 个孔的 p_z 完全相同 -> p 全在一条线上 -> σ2 = 0
    print("  单高度时 4 个相机点：")
    for q in p1:
        print("    ({:8.3f}, {:8.3f}, {:8.3f})".format(*q))
    print("  双高度时 8 个相机点：")
    for q in p2:
        print("    ({:8.3f}, {:8.3f}, {:8.3f})".format(*q))
    print("  -> 单高度 σ2≈0（共线，绕这条线转任意角都最优）；双高度把点撑成 2D")

    p2n, P2n, A2n = make_dataset(noise=0.05, levels=Z_LEVELS, seed=3)
    r2n = report("双高度 噪声0.05", p2n, P2n, A2n)

    assert r1["ratio"] < 1e-3 and r1["r_err"] > 1.0      # 单高度退化
    assert r2["ratio"] > 0.02 and r2["r_err"] < 1e-3     # 双高度唯一解
    assert r2n["r_err"] < 0.05


def demo_transpose():
    print("\n三、H 转置陷阱：R 解成 R^T，矩阵「看着差不多」，方向反了")
    p, P, A = make_dataset(noise=0.0, seed=4)
    q = np.array([(inv_T(a) @ np.append(Pi, 1.0))[:3] for Pi, a in zip(P, A)])

    R_ok, t_ok, _, _ = kabsch(p, q)                      # 正确配对

    # 故意把 H 写成 Σ q̃ p̃ᵀ，却仍用 R = V diag Uᵀ
    p_bar, q_bar = p.mean(0), q.mean(0)
    H_wrong = (q - q_bar).T @ (p - p_bar)
    U, S, Vt = np.linalg.svd(H_wrong)
    V = Vt.T
    d = float(np.sign(np.linalg.det(V @ U.T)))
    R_bad = V @ np.diag([1.0, 1.0, d]) @ U.T

    print("  正确解 vs 真值：旋转误差 {:.4f}°".format(rot_error_deg(R_ok, X_TRUE[:3, :3])))
    print("  转置解 vs R_true^T：旋转误差 {:.4f}°".format(rot_error_deg(R_bad, X_TRUE[:3, :3].T)))
    # 残差也不一定大 —— 这正是它危险的地方
    pred_ok = np.array([(a @ make_T(R_ok, t_ok) @ np.append(pi, 1.0))[:3] for pi, a in zip(p, A)])
    pred_bad = np.array([(a @ make_T(R_bad, t_ok) @ np.append(pi, 1.0))[:3] for pi, a in zip(p, A)])
    print("  用正确 t 时：正确解 RMS={:.3f} mm，转置解 RMS={:.3f} mm".format(
        np.sqrt(np.mean(np.sum((pred_ok - P) ** 2, axis=1))),
        np.sqrt(np.mean(np.sum((pred_bad - P) ** 2, axis=1)))))

    assert rot_error_deg(R_bad, X_TRUE[:3, :3].T) < 1e-6
    assert rot_error_deg(R_bad, R_ok) > 0.5


def demo_pure_translation_axxb():
    print("\n四、同一个「纯平移」，对 AX=XB 是致命的，对本项目不是")
    # 本项目：两次扫描只差平移（现场就是这样：Z2 扫完抬到 Z1）
    p, P, A = make_dataset(noise=0.0, seed=5)
    is_same_rot = np.allclose(A[0][:3, :3], A[4][:3, :3])
    print("  两次扫描的姿态差：{}".format("单位阵（纯平移）" if is_same_rot else "有旋转"))
    r = report("纯平移（本项目）", p, P, A)

    # 经典 AX=XB：A_ij·X = X·B_ij，A_ij 是纯平移 I|a，B_ij 是相机看到的同一运动
    a_vec = np.array([0.0, 0.0, 40.0])
    A_ij = make_T(np.eye(3), a_vec)
    B_ij = inv_T(X_TRUE) @ A_ij @ X_TRUE

    k = a_vec / np.linalg.norm(a_vec)
    # 假 X1：绕平移轴（激光线/运动方向）转 5° —— 完全不可观
    X_about = make_T(rodrigues(np.radians(5.0) * k) @ X_TRUE[:3, :3], X_TRUE[:3, 3])
    # 假 X2：绕垂直平移轴的轴转 5° —— 只有这一个方向能被约束到
    X_perp = make_T(rodrigues(np.radians([5.0, 0.0, 0.0])) @ X_TRUE[:3, :3], X_TRUE[:3, 3])

    def axxb_resid(X):
        return float(np.linalg.norm(A_ij @ X - X @ B_ij))

    print("  AX=XB 残差：真值 X = {:.2e} mm".format(axxb_resid(X_TRUE)))
    print("  AX=XB 残差：绕平移轴差 5° = {:.2e} mm  <- 完全不敏感".format(axxb_resid(X_about)))
    print("  AX=XB 残差：绕垂直轴差 5° = {:.2f} mm  <- 只有一个方向有约束".format(axxb_resid(X_perp)))

    # 同一组数据换成"对应点已知"，5° 误差立刻暴露
    pred = np.array([(a @ X_about @ np.append(pi, 1.0))[:3] for pi, a in zip(p, A)])
    rms_about = float(np.sqrt(np.mean(np.sum((pred - P) ** 2, axis=1))))
    print("  本项目（对应点已知）：同一个 5° 误差 -> RMS = {:.2f} mm".format(rms_about))
    print("  -> 信息来自 8 个点的空间分布，不是来自机器人姿态差异")

    assert r["r_err"] < 1e-6                          # 纯平移仍能解出手眼旋转
    assert axxb_resid(X_about) < 1e-9                 # AX=XB 对绕平移轴的旋转零敏感
    assert axxb_resid(X_perp) > 1.0                   # 垂直方向才有约束
    assert rms_about > 5.0


def demo_collinear():
    print("\n五、真正的退化：圆心在传感器平面里共线（σ2/σ1 -> 0，残差小但解错）")
    # 所有孔排在同一条母线上（x 相同），即使扫两个高度，相机点也只有 2 个不同值
    line_pts = [(0.0, -150.0), (0.0, -50.0), (0.0, 50.0), (0.0, 150.0)]
    p, P, A = make_dataset(noise=0.0, circles=line_pts, seed=6)
    r = report("共线（4 圆 x 2 高度）", p, P, A)
    p2, P2, A2 = make_dataset(noise=0.0, seed=6)
    r2 = report("张开（4 圆 x 2 高度）", p2, P2, A2)

    print("  共线时残差 {:.2e} mm，但旋转误差 {:.3f}° —— 「残差小 ≠ 解唯一」".format(r["rms"], r["r_err"]))
    assert r["ratio"] < 1e-3 and r["r_err"] > 1.0
    assert r2["ratio"] > 0.02 and r2["r_err"] < 1e-3


def demo_noise_sweep():
    print("\n六、噪声扫描：0.005 -> 0.2 mm，误差怎么涨")
    print("  噪声(mm) |    RMS(mm) | 旋转误差(°) | 平移误差(mm)")
    for sigma in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        p, P, A = make_dataset(noise=sigma, seed=7)
        r = report("噪声 σ={:.3f}".format(sigma), p, P, A)


def main():
    demo_basic()
    demo_heights()
    demo_transpose()
    demo_pure_translation_axxb()
    demo_collinear()
    demo_noise_sweep()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
