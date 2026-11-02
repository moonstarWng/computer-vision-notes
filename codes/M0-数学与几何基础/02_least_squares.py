# -*- coding: utf-8 -*-
"""
M0 第 2 篇 · 最小二乘 -> 高斯牛顿 -> LM -> 鲁棒核
对应笔记：notes/M0-数学与几何基础/2-最小二乘与LM.md（§2~§7）

这个脚本回答四个问题
--------------------
1. "最小二乘"到底在最小化什么？为什么能一步解出来？（线性情况）
2. 非线性时怎么办？高斯牛顿（GN）那一步 (J^T J)δ = -J^T r 是怎么来的？
3. GN 什么时候会炸？LM 的 λ 是怎么救场的？
4. 有一个离群点时，为什么鲁棒核能让结果更准？

运行
----
    python 02_least_squares.py

场景：拟合 y = a·exp(b·x)，数据里故意放 3 个离群点。
"""

import numpy as np


# =====================================================================
# 一、线性最小二乘：有闭式解
# =====================================================================

def demo_linear():
    """
    线性问题：已知 A（50×3）和 b（50），求 x 使 ‖Ax − b‖² 最小。

    正规方程（对 x 求导令 0）：
        A^T A · x = A^T b

    注意 A 是"高瘦"的（50 行 3 列），方程数比未知数多，
    一般没有精确解，只能找"残差平方和最小"的那个 x。
    """
    print("== 1. 线性最小二乘：正规方程 vs lstsq ==")
    rng = np.random.default_rng(0)                  # 固定种子，结果可复现

    A = rng.normal(size=(50, 3))                    # 50 个方程、3 个未知数
    x_true = np.array([1.0, -2.0, 0.5])             # 真值（我们心里知道）
    b = A @ x_true + rng.normal(0, 0.05, 50)        # 观测 = 真值 + 噪声

    # 正规方程：解 3×3 的线性方程组
    x_ne = np.linalg.solve(A.T @ A, A.T @ b)
    # lstsq 内部用 QR 分解，数值更稳（SLAM 里也推荐用它做小规模求解）
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"正规方程 : {x_ne.round(4)}")
    print(f"lstsq    : {x_ls.round(4)}")
    print(f"两者差   : {np.abs(x_ne - x_ls).max():.2e}\n")

    # -----------------------------------------------------------------
    print("== 2. 秩亏的例子（两列共线） ==")
    # 这里的两个参数 a、b 完全不可区分：第二列 = 2 × 第一列
    # 所以 A^T A 秩为 1，方程没有唯一解。
    x = np.linspace(0, 2, 40)
    A2 = np.column_stack([x, 2 * x])
    b2 = 3 * x
    print(f"det(A^T A) = {np.linalg.det(A2.T @ A2):.1f}")   # 0，说明不可逆
    try:
        np.linalg.solve(A2.T @ A2, A2.T @ b2)
        print("solve 居然成功了？")
    except np.linalg.LinAlgError as e:
        # 这就是工程里"参数不可观"时的典型报错
        print(f"solve -> LinAlgError: {e}")
    # lstsq 会给出"最小范数解"：在 a + 2b = 3 的所有解里挑长度最小的
    print(f"lstsq -> {np.linalg.lstsq(A2, b2, rcond=None)[0].round(4)}"
          f"  （a+2b=3 的最小范数解）\n")


# =====================================================================
# 二、非线性最小二乘：GN 和 LM
# =====================================================================

def make_data(seed=0):
    """
    造一批带离群点的数据。

    真模型：y = 2·exp(0.5x)
    噪声  ：σ = 0.05 的高斯噪声
    离群点：第 5、20、33 个点额外加了 2.0 / -2.5 / 3.0
            （模拟误匹配、误检测这类"错得离谱"的观测）
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2, 40)
    y_true = 2.0 * np.exp(0.5 * x)
    y = y_true + rng.normal(0, 0.05, x.size)
    y[[5, 20, 33]] += np.array([2.0, -2.5, 3.0])
    return x, y, y_true


def residual(p, x, y):
    """
    残差：预测值 − 观测值

        r_i(p) = a·exp(b·x_i) − y_i          p = (a, b)

    注意残差是"向量"（每个点一个），不是标量。
    优化目标是最小化 ½·‖r‖² = ½ Σ r_i²。
    """
    a, b = p
    return a * np.exp(b * x) - y


def jacobian(p, x):
    """
    雅可比矩阵 J = ∂r/∂p，形状 (m, 2)：m 个残差、2 个参数。

    对残差 r_i = a·exp(b·x_i) − y_i 求偏导：
        ∂r_i/∂a = exp(b·x_i)
        ∂r_i/∂b = a·x_i·exp(b·x_i)

    每一行就是"第 i 个点的残差对 (a,b) 有多敏感"。
    """
    a, b = p
    e = np.exp(b * x)
    # column_stack 把两列拼起来：第 1 列 ∂r/∂a，第 2 列 ∂r/∂b
    return np.column_stack([e, a * x * e])


def gauss_newton(p, x, y, iters=50):
    """
    高斯牛顿法

    每一步做三件事：
        1) 在当前 p 处算残差 r 和雅可比 J；
        2) 解线性方程 (J^T J) δ = -J^T r；
        3) p <- p + δ，重复。

    为什么是这个方程？
        把 r(p+δ) 一阶展开：r + Jδ
        目标 ½‖r + Jδ‖² 对 δ 求导令 0，就得到 J^T(r + Jδ) = 0。

    J^T J 是 Hessian 的近似（丢掉了二阶导项），所以叫"高斯-牛顿"。
    它不检查这一步是否变好，算完就走 —— 这是它容易发散的原因。
    """
    for k in range(iters):
        r = residual(p, x, y)                   # 当前的 m 维残差
        J = jacobian(p, x)                      # 当前的 m×2 雅可比
        # solve 比 inv(...) 快且稳，永远不要写 np.linalg.inv(J.T @ J) @ ...
        dp = np.linalg.solve(J.T @ J, -J.T @ r)
        p = p + dp
        if np.linalg.norm(dp) < 1e-10:          # 参数基本不动了，收敛
            break
    return p, k                                 # k 是迭代次数（0 开始计数）


def levenberg_marquardt(p, x, y, lam=1e-3, iters=300):
    """
    LM（Levenberg-Marquardt）：给高斯牛顿装了"刹车"

        (J^T J + λ·diag(J^T J)) δ = -J^T r

    λ 的作用：
        λ -> 0   退化回 GN（快，但可能飞出去）
        λ -> ∞   退化成很小的梯度下降步（稳，但慢）

    为什么只加在对角？
        参数尺度不同（像素 / 米 / 弧度），diag(J^T J) 正好是每个参数的
        "量纲"，乘上 λ 后不会把某个参数压得过狠。

    λ 的自适应规则：
        这一步让 cost 变小  -> 接受，λ ← λ/3   （更信任 GN）
        这一步让 cost 变大  -> 拒绝，λ ← λ×5   （收小步子再试）
    """
    r = residual(p, x, y)
    cost = r @ r                                # 当前代价（残差平方和）
    for k in range(iters):
        J = jacobian(p, x)
        H = J.T @ J
        A = H + lam * np.diag(np.diag(H))       # 只在对角线上加阻尼
        dp = np.linalg.solve(A, -J.T @ r)

        r_new = residual(p + dp, x, y)
        cost_new = r_new @ r_new
        if cost_new < cost:                     # 变好：接受这一步
            p, r, cost = p + dp, r_new, cost_new
            lam = max(lam / 3.0, 1e-12)         # 放开，下次更大胆
            if np.linalg.norm(dp) < 1e-10:
                break
        else:                                   # 变差：拒绝这一步
            lam *= 5.0                          # 收紧，下次更保守
    return p, k, lam


# =====================================================================
# 三、鲁棒核：让离群点"闭嘴"
# =====================================================================

def huber_weight(r, delta=0.5):
    """
    Huber 损失的权重函数（IRLS 用）

        ρ(r) = r²/2                |r| ≤ δ
               δ(|r| − δ/2)         |r| > δ
        权重 w = ρ'(r)/r =
               1                   |r| ≤ δ
               δ / |r|             |r| > δ

    含义：残差小时当普通最小二乘（权重 1）；残差大时权重按 1/|r| 衰减，
    所以离群点对解的影响被压住了。

    np.where(条件, 满足时的值, 不满足时的值) 相当于向量版的 if-else。
    """
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def irls_huber(p, x, y, delta=0.5, iters=100):
    """
    IRLS（迭代重加权最小二乘）：鲁棒核的标准工程实现

    每一轮：
        1) 用当前残差算权重 w_i = ρ'(r_i)/r_i；
        2) 解加权最小二乘 (J^T W J) δ = -J^T W r；
        3) 更新参数，重复。

    这就是"把鲁棒问题变成一串普通最小二乘"的经典技巧。
    """
    for k in range(iters):
        r = residual(p, x, y)
        w = huber_weight(r, delta)              # 每个点一个权重，形状 (m,)
        J = jacobian(p, x)
        # w[:, None] 把 (m,) 变成 (m,1)，这样才能和 (m,2) 的 J 逐行相乘
        H = J.T @ (w[:, None] * J)              # J^T W J
        g = J.T @ (w * r)                       # J^T W r
        dp = np.linalg.solve(H, -g)
        p = p + dp
        if np.linalg.norm(dp) < 1e-12:
            break
    return p, k


def inlier_rmse(p, x, y, mask):
    """
    只在"内点"上算 RMSE（均方根误差）。

    为什么要分开算？
        有离群点时，把所有点混在一起算 RMSE 会被巨大的残差主导，
        看不出算法到底把正常点拟合得多好。
    """
    r = residual(p, x, y)
    return float(np.sqrt(np.mean(r[mask] ** 2)))   # mask=True 的位置参与平均


# =====================================================================
# 四、主流程
# =====================================================================

def demo_nonlinear():
    x, y, _ = make_data()

    mask = np.ones(x.size, bool)                # 全 True
    mask[[5, 20, 33]] = False                   # 标记出 3 个离群点（False）
    p0 = np.array([1.0, 0.0])                   # 初始猜测 a=1, b=0

    p_gn, it_gn = gauss_newton(p0.copy(), x, y)
    p_lm, it_lm, lam = levenberg_marquardt(p0.copy(), x, y)
    p_hu, it_hu = irls_huber(p0.copy(), x, y)

    print("== 3. 有离群点时：GN vs LM vs Huber ==")
    print(f"{'方法':<8}{'a':>8}{'b':>8}{'内点RMSE':>12}{'迭代':>6}")
    for name, p, it in [("GN", p_gn, it_gn), ("LM", p_lm, it_lm), ("Huber", p_hu, it_hu)]:
        print(f"{name:<8}{p[0]:>8.4f}{p[1]:>8.4f}{inlier_rmse(p, x, y, mask):>12.4f}{it:>6}")
    print(f"{'真值':<8}{2.0:>8.4f}{0.5:>8.4f}{'-':>12}{'-':>6}")
    print(f"（LM 结束时 λ = {lam:.2e}）\n")

    # 核心结论：Huber 的内点残差应该明显小于 GN
    # 如果不是，说明实现有问题（这个断言就是防你自己改错）
    assert inlier_rmse(p_hu, x, y, mask) < inlier_rmse(p_gn, x, y, mask), \
        "预期 Huber 比 GN 更准，检查实现"
    print("== 结论：Huber 的总 L2 可能更大，但内点残差更小 ==")
    print("   有离群点时，用 L2 的总和评价算法是错的。\n")


def main():
    demo_linear()
    demo_nonlinear()
    print("ALL DONE")


if __name__ == "__main__":
    main()
