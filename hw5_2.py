# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:55:37 2025

@author: ericd
"""

import numpy as np

# 定義右邊的函數 f(t, u)
def f(t, u):
    u1, u2 = u
    f1 = 9 * u1 + 24 * u2 + 5 * np.cos(t) - (1/3) * np.sin(t)
    f2 = -24 * u1 - 52 * u2 - 9 * np.cos(t) + (1/3) * np.sin(t)
    return np.array([f1, f2])

# 精確解
def exact_solution(t):
    u1 = 2 * np.exp(-3 * t) - np.exp(-39 * t) + (1/3) * np.cos(t)
    u2 = -np.exp(-3 * t) + 2 * np.exp(-39 * t) - (1/3) * np.cos(t)
    return np.array([u1, u2])

# RK4 方法
def rk4_method(h, t_end):
    t = np.arange(0, t_end + h, h)
    n = len(t)
    u = np.zeros((n, 2))  # u[:, 0] 是 u1, u[:, 1] 是 u2
    u[0] = [4/3, -2/3]   # 初始條件
    
    for i in range(n-1):
        k1 = f(t[i], u[i])
        k2 = f(t[i] + h/2, u[i] + (h/2) * k1)
        k3 = f(t[i] + h/2, u[i] + (h/2) * k2)
        k4 = f(t[i] + h, u[i] + h * k3)
        u[i+1] = u[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, u

# 計算不同步長的結果
t_end = 1.0
h_values = [0.05, 0.1]

for h in h_values:
    print(f"\n步長 h = {h}:")
    t, u = rk4_method(h, t_end)
    
    # 計算精確解
    u_exact = np.array([exact_solution(ti) for ti in t])
    
    # 比較結果（只顯示 t = 0, 0.5, 1 的值）
    indices = [0, int(0.5/h), int(1.0/h)]
    for idx in indices:
        print(f"t = {t[idx]:.1f}:")
        print(f"  數值解: u1 = {u[idx, 0]:.6f}, u2 = {u[idx, 1]:.6f}")
        print(f"  精確解: u1 = {u_exact[idx, 0]:.6f}, u2 = {u_exact[idx, 1]:.6f}")
        print(f"  誤差:    u1 = {abs(u[idx, 0] - u_exact[idx, 0]):.6f}, u2 = {abs(u[idx, 1] - u_exact[idx, 1]):.6f}")