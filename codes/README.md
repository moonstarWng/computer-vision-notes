# 配套示例代码

学习笔记里出现的代码，在这里是**可直接运行、带断言**的版本。
每篇笔记 → 一个脚本，函数名与笔记一致，方便对照。

## 目录

```
codes/
├── requirements.txt
└── M0-数学与几何基础/
    ├── 01_rotation.py           # 旋转：hat/vee、exp/log、四元数、det 陷阱
    ├── 02_least_squares.py      # 正规方程、GN、LM、Huber + IRLS
    ├── 03_kabsch.py             # SVD 求 R,t、det 修正、共线退化
    └── 04_kalman.py             # 误差传播、高斯融合、KF、马氏距离
```

## 环境

```powershell
cd codes
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

依赖只有 `numpy` + `scipy`（scipy 仅用于 01 里的互转对照，没装也能跑，会跳过那一节）。

## 运行

```powershell
python M0-数学与几何基础/01_rotation.py
python M0-数学与几何基础/02_least_squares.py
python M0-数学与几何基础/03_kabsch.py
python M0-数学与几何基础/04_kalman.py
```

## 脚本 ↔ 笔记对照

| 脚本 | 笔记 | 关键输出（会对上的数字） |
|---|---|---|
| `01_rotation.py` | [`1-旋转.md`](../notes/M0-数学与几何基础/1-旋转.md) | exp/log 往返误差 `4.4e-11`；det=1 但 `RᵀR≠I` |
| `02_least_squares.py` | [`2-最小二乘与LM.md`](../notes/M0-数学与几何基础/2-最小二乘与LM.md) | GN 内点 RMSE `0.0903` vs Huber `0.0420` |
| `03_kabsch.py` | [`3-SVD求刚体变换.md`](../notes/M0-数学与几何基础/3-SVD求刚体变换.md) | 镜像数据：修正后 det `+1` 残差 `6.962`，不修正 det `-1` 残差 `1.146` |
| `04_kalman.py` | [`4-协方差传播与卡尔曼.md`](../notes/M0-数学与几何基础/4-协方差传播与卡尔曼.md) | 传播 vs 蒙特卡洛误差 `0.4%`；KF 收敛到 `[5.063, 1.015]` |

## 约定

- **无外部数据**：全部随机生成 + 固定种子，任何机器上结果一致；
- **带断言**：脚本末尾会打印 `ALL PASS` / `ALL DONE`，失败会直接报错；
- **纯函数 + `main()`**：函数可以单独 import 到别的脚本里复用；
- **注释对着笔记写**：公式旁边的注释就是笔记里的那句话。

## 后续

- 每学完一篇新笔记，就在这里加一个同编号脚本；
- 有真实数据的实验（棋盘格、点云）放到 `labs/`，这里只放**最小可运行示例**。
