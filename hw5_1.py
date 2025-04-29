# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:41:25 2025

@author: ericd
"""

import numpy as np
import matplotlib.pyplot as plt

# 定義微分方程 f(t, y) = 1 + y/t + (y/t)^2
def f(t, y):
    return 1 + y/t + (y/t)**2

# 精確解 y(t) = t * tan(ln(t))
def exact_solution(t):
    return t * np.tan(np.log(t))

# Euler 方法
def euler_method(t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        y_values[i] = y + h * f(t, y)
    
    return t_values, y_values

# 二階泰勒方法需要的導數 df/dt
def df_dt(t, y):
    # f(t, y) = 1 + y/t + (y/t)^2
    # 計算 df/dt = ∂f/∂t + f * ∂f/∂y
    partial_f_t = -y/t**2 - 2*y**2/t**3  # ∂f/∂t = -y/t^2 - 2y^2/t^3
    partial_f_y = 1/t + 2*y/t**2         # ∂f/∂y = 1/t + 2y/t^2
    return partial_f_t + f(t, y) * partial_f_y

# 二階泰勒方法
def taylor_order2(t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        T2 = f(t, y) + (h/2) * df_dt(t, y)  # T^(2) = f + (h/2) * df/dt
        y_values[i] = y + h * T2
    
    return t_values, y_values

# 參數
t0, y0 = 1.0, 0.0  # 初始條件
t_end = 2.0        # 結束時間
h = 0.1            # 步長

# 計算數值解
t_euler, y_euler = euler_method(t0, y0, t_end, h)
t_taylor, y_taylor = taylor_order2(t0, y0, t_end, h)

# 計算精確解
y_exact = exact_solution(t_euler)

# 計算誤差
error_euler = np.abs(y_exact - y_euler)
error_taylor = np.abs(y_exact - y_taylor)

# 輸出結果
print("t\tExact\t\tEuler\t\tEuler Error\t\tTaylor\tTaylor Error")
print("-" * 70)
for i in range(len(t_euler)):
    print(f"{t_euler[i]:.1f}\t{y_exact[i]:.6f}\t{y_euler[i]:.6f}\t{error_euler[i]:.6f}\t{y_taylor[i]:.6f}\t{error_taylor[i]:.6f}")

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(t_euler, y_exact, 'k-', label='Exact Solution')
plt.plot(t_euler, y_euler, 'ro-', label='Euler Method')
plt.plot(t_euler, y_taylor, 'b^-', label='Taylor Order 2')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Numerical Solutions vs Exact Solution')
plt.legend()
plt.grid(True)
plt.savefig('ivp_numerical_solutions.png')
plt.close()