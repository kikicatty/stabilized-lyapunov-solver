import numpy as np
import lyapunov_solver as ly
# A = np.array([[1, 2], [3, 4]])  # 非对称实矩阵[^4]

# Q = np.eye(2)

# X = ly.solve_lyapunov(A, Q)
# print(f"残差范数:{np.linalg.norm(A@X + X@A.T + Q)}")
n=10
A=np.random.randn(n,n)
A=A.T@A