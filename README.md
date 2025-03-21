# Stabilized Lyapunov Solver 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)

面向非线性控制系统的高性能Lyapunov方程求解器（源自数学院硕士课题）

<img src="docs/iter_visual.png" alt="迭代收敛图示" width="600">

## 核心特征
- **数值稳定算法**：采用Schur分解与块状矩阵处理技术
- **降阶加速**：集成Krylov子空间方法加速大规模计算


## 开始使用
```bash
git clone https://github.com/kikicatty/stabilized-lyapunov-solver.git
conda install -c conda-forge numpy scipy cupy
```

## 使用示例（类比电商数据分析项目展示方式[^4]）
```python
from solver import schur_solve
import numpy as np

# 生成非对称系统矩阵
A = np.random.randn(500,500)  
Q = np.eye(500)

# 数值稳定解法
X = schur_solve(A, Q, epsilon=1e-6)  
residual = np.linalg.norm(A@X + X@A.T + Q)
print(f"残差范数: {residual:.2e}")
```
