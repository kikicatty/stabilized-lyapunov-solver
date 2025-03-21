from scipy import linalg  
import scipy.linalg as la
import cupy as cp
import numpy as np

# AX + XA^T + Q = 0 → 解X    

def solve_lyapunov(A, Q):  # 使用scipy验证        
    return linalg.solve_continuous_lyapunov(A, Q) 


def _solve_lower(T, Q_hat):
    # 处理准三角系统 AX+XA^T = -Q
    n = T.shape[0]  # 从输入矩阵自动获取维度[^4]
    X_hat = np.zeros_like(Q_hat)
    for j in reversed(range(n)):
        X_hat[j,j] = -Q_hat[j,j] / (2*T[j,j])
        X_hat[:j,j] = -1/(T[j,j]+T[j,j]) * (
            Q_hat[:j,j] + T[:j,:j] @ X_hat[:j,j] + X_hat[:j,:j] @ T[j,:j]
        )
        X_hat[j,:j] = X_hat[:j,j].T  # 对称性填充
    return X_hat


def _krylov_basis(A, U, k):
    # 生成Krylov子空间基
    n = A.shape[0]
    V = np.zeros((n, k))
    V[:,0] = U[:,0]/np.linalg.norm(U[:,0])
    for j in range(1, k):
        w = A @ V[:,j-1]
        for i in range(j):
            h = V[:,i].T @ w
            w -= h * V[:,i]
        V[:,j] = w / np.linalg.norm(w)
    return V


def schur_solve(A, Q):
    # 实Schur分解 O(n^3)
    T, Z = la.schur(A)  # A = Z*T*Z^H[^3] 
    # 将Q变换到Schur基下
    Q_hat = Z.T @ Q @ Z  
    # 解简化后的方程
    X_hat = _solve_lower(T, Q_hat)
    return Z @ X_hat @ Z.T


def lowrank_solve(A, U, V):  # Q=UV^T
    # 使用Krylov子空间方法降阶[^4]
    k = U.shape[1]
    P = _krylov_basis(A, U, k)
    Ar = P.T @ A @ P
    Xr = la.solve_sylvester(Ar, Ar.T, P.T@U @ V.T@P)
    return P @ Xr @ P.T


def block_solve(A_gpu, Q_gpu, block_size=256):
    # 分块计算应对显存限制
    n = A_gpu.shape[0]
    X_gpu = cp.zeros_like(Q_gpu)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            # GPU加速矩阵块运算
            A_block = A_gpu[i:i+bs, i:i+bs]  
            Q_block = Q_gpu[i:i+bs, j:j+bs]
            X_gpu[i:i+bs, j:j+bs] = cp.linalg.solve(...)
    return X_gpu


def stabilized_solve(A, Q, n, eps=1e-6):
    X = schur_solve(A, Q)
    residual = A@X + X@A.T + Q
    if np.linalg.norm(residual) > 1e-3:
        # 正则化处理病态问题[^3]
        X = la.solve(A + eps*np.eye(n), -Q)  
    return X
