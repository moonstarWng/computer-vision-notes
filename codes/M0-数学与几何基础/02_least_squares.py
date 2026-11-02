"""
M0 第 2 篇 · 最小二乘 → 高斯牛顿 → LM → 鲁棒核
对应笔记：notes/M0-数学与几何基础/2-最小二乘与LM.md（§2、§3、§4、§6、§7）

运行：
    python 02_least_squares.py

场景：拟合 y = a·exp(b·x)，数据里故意放 3 个离群点。
"""

import numpy as np


# ---------------------------------------------------------------- 线性最小二乘
def demo_linear():
    print("== 1. 线性最小二乘：正规方程 vs lstsq ==")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3))
    x_true = np.array([1.0, -2.0, 0.5])
    b = A @ x_true + rng.normal(0, 0.05, 50)

    x_ne = np.linalg.solve(A.T @ A, A.T @ b)   # 正规方程
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"正规方程 : {x_ne.round(4)}")
    print(f"lstsq    : {x_ls.round(4)}")
    print(f"两者差   : {np.abs(x_ne - x_ls).max():.2e}\n")

    print("== 2. 秩亏的例子（两列共线） ==")
    x = np.linspace(0, 2, 40)
    A2 = np.column_stack([x, 2 * x])           # 第二列 = 2 × 第一列
    b2 = 3 * x
    print(f"det(A^TA) = {np.linalg.det(A2.T @ A2):.1f}")
    try:
        np.linalg.solve(A2.T @ A2, A2.T @ b2)
        print("solve 居然成功了？")
    except np.linalg.LinAlgError as e:
        print(f"solve -> LinAlgError: {e}")
    print(f"lstsq -> {np.linalg.lstsq(A2, b2, rcond=None)[0].round(4)}"
          f"  （a+2b=3 的最小范数解）\n")


# ---------------------------------------------------------------- 非线性问题
def make_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2, 40)
    y_true = 2.0 * np.exp(0.5 * x)
    y = y_true + rng.normal(0, 0.05, x.size)
    y[[5, 20, 33]] += np.array([2.0, -2.5, 3.0])   # 3 个离群点
    return x, y, y_true


def residual(p, x, y):
    a, b = p
    return a * np.exp(b * x) - y


def jacobian(p, x):
    a, b = p
    e = np.exp(b * x)
    return np.column_stack([e, a * x * e])         # [dr/da, dr/db]


def gauss_newton(p, x, y, iters=50):
    for k in range(iters):
        r = residual(p, x, y)
        J = jacobian(p, x)
        dp = np.linalg.solve(J.T @ J, -J.T @ r)    # (J^TJ)δ = -J^Tr
        p = p + dp
        if np.linalg.norm(dp) < 1e-10:
            break
    return p, k


def levenberg_marquardt(p, x, y, lam=1e-3, iters=300):
    r = residual(p, x, y)
    cost = r @ r
    for k in range(iters):
        J = jacobian(p, x)
        A = J.T @ J + lam * np.diag(np.diag(J.T @ J))   # 只在对角加阻尼
        dp = np.linalg.solve(A, -J.T @ r)
        r_new = residual(p + dp, x, y)
        cost_new = r_new @ r_new
        if cost_new < cost:                             # 变好：接受，放开 λ
            p, r, cost = p + dp, r_new, cost_new
            lam = max(lam / 3.0, 1e-12)
            if np.linalg.norm(dp) < 1e-10:
                break
        else:                                           # 变差：拒绝，收紧 λ
            lam *= 5.0
    return p, k, lam


def huber_weight(r, delta=0.5):
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def irls_huber(p, x, y, delta=0.5, iters=100):
    """Huber + 迭代重加权最小二乘（IRLS）"""
    for k in range(iters):
        r = residual(p, x, y)
        w = huber_weight(r, delta)
        J = jacobian(p, x)
        H = J.T @ (w[:, None] * J)       # J^TWJ
        g = J.T @ (w * r)                # J^TWr
        dp = np.linalg.solve(H, -g)
        p = p + dp
        if np.linalg.norm(dp) < 1e-12:
            break
    return p, k


def inlier_rmse(p, x, y, mask):
    r = residual(p, x, y)
    return float(np.sqrt(np.mean(r[mask] ** 2)))


def demo_nonlinear():
    x, y, _ = make_data()
    mask = np.ones(x.size, bool)
    mask[[5, 20, 33]] = False          # 内点
    p0 = np.array([1.0, 0.0])

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
