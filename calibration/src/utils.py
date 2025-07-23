#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def tri2F6(A_3d, B_3d, C_3d):
    """
    从正三角形的三个顶点构建F6变换参数
    
    坐标系定义：
    - 原点：A顶点
    - X轴：AB方向
    - Z轴：AB × AC方向（右手法则）
    - Y轴：Z × X方向
    
    参数：
        A_3d, B_3d, C_3d: 三角形三个顶点的3D坐标 (numpy array)
    
    返回：
        f6: [tx, ty, tz, rx, ry, rz] - F6变换参数
    """
    # 确保输入是numpy数组
    A = np.array(A_3d, dtype=float)
    B = np.array(B_3d, dtype=float)
    C = np.array(C_3d, dtype=float)
    
    # 构建坐标系
    origin = A
    
    # X轴：AB方向
    AB = B - A
    X_axis = AB / np.linalg.norm(AB)
    
    # Z轴：AB × AC方向（右手法则）
    AC = C - A
    Z_axis_unnorm = np.cross(AB, AC)
    Z_axis = Z_axis_unnorm / np.linalg.norm(Z_axis_unnorm)
    
    # Y轴：Z × X方向
    Y_axis = np.cross(Z_axis, X_axis)
    
    # 构建旋转矩阵
    R = np.column_stack([X_axis, Y_axis, Z_axis])
    
    # 验证旋转矩阵的正交性
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        print(f"警告：旋转矩阵行列式 = {np.linalg.det(R):.6f}，应该接近1.0")
    
    # 平移向量
    t = origin
    
    # 将旋转矩阵转换为轴角表示
    rotation_vector = r2o(R)
    
    # 构建F6参数
    f6 = np.concatenate([t, rotation_vector])
    
    return f6

def r2o(R):
    """
    将旋转矩阵转换为轴角表示
    
    参数：
        R: 3×3旋转矩阵
    
    返回：
        axis_angle: 3维轴角向量，方向为旋转轴，模长为旋转角度（度）
    """
    # 使用罗德里格斯公式的逆变换
    trace = np.trace(R)
    
    # 计算旋转角度（弧度）
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # 限制在[-1,1]范围内
    theta_rad = np.arccos(cos_theta)
    
    if theta_rad < 1e-6:
        # 接近单位矩阵，无旋转
        return np.array([0.0, 0.0, 0.0])
    elif abs(theta_rad - np.pi) < 1e-6:
        # 180度旋转的特殊情况
        # 找到特征值为1的特征向量
        eigenvals, eigenvecs = np.linalg.eig(R)
        real_eigenvals = np.real(eigenvals)
        idx = np.argmin(np.abs(real_eigenvals - 1.0))
        axis = np.real(eigenvecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        # 转换为度数
        theta_deg = np.rad2deg(theta_rad)
        return theta_deg * axis
    else:
        # 一般情况
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0], 
            R[1, 0] - R[0, 1]
        ])
        axis = axis / (2 * np.sin(theta_rad))
        # 转换为度数
        theta_deg = np.rad2deg(theta_rad)
        return theta_deg * axis

def o2r(axis_angle):
    """
    轴角表示转换为旋转矩阵（输入为度数单位）
    
    参数：
        axis_angle: 3维轴角向量，角度单位为度
    
    返回：
        R: 3×3旋转矩阵
    """
    theta_deg = np.linalg.norm(axis_angle)
    
    if theta_deg < 1e-6:
        return np.eye(3)
    
    # 转换为弧度
    theta_rad = np.deg2rad(theta_deg)
    
    # 单位轴向量
    k = axis_angle / theta_deg
    
    # 罗德里格斯公式
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    R = np.eye(3) + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * np.dot(K, K)
    
    return R

def apply_F6_transform(point_local, f6):
    """
    应用F6变换，将局部坐标转换为全局坐标
    
    使用F6群内运算：将点表示为特殊的F6（纯平移，无旋转）
    
    参数：
        point_local: 局部坐标系中的点 [x, y, z]
        f6: F6变换参数 [tx, ty, tz, rx, ry, rz]
    
    返回：
        point_global: 全局坐标系中的点
    """
    # 将点转换为F6表示：平移分量为点坐标，旋转分量为零
    f6_point = np.concatenate([np.array(point_local), np.array([0.0, 0.0, 0.0])])
    
    # 使用F6群运算：f6 @ f6_point
    # 先计算旋转矩阵
    R = o2r(f6[3:6])
    
    # F6复合运算的结果
    # 新的平移 = f6的平移 + f6的旋转应用于点
    point_rotated = np.dot(R, f6_point[0:3])
    point_global = f6[0:3] + point_rotated
    
    return point_global

def validate_triangle(A_3d, B_3d, C_3d, expected_edge_length, tolerance=1.0):
    """
    验证三角形的几何属性
    
    参数：
        A_3d, B_3d, C_3d: 三角形顶点坐标
        expected_edge_length: 期望的边长（mm）
        tolerance: 允许的误差（mm）
    
    返回：
        is_valid: 是否为有效的等边三角形
        edge_lengths: 三条边的长度 [AB, BC, CA]
        edge_errors: 边长误差 [AB_err, BC_err, CA_err]
    """
    A = np.array(A_3d)
    B = np.array(B_3d)
    C = np.array(C_3d)
    
    # 计算边长
    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    CA = np.linalg.norm(A - C)
    
    edge_lengths = [AB, BC, CA]
    edge_errors = [abs(AB - expected_edge_length), 
                   abs(BC - expected_edge_length), 
                   abs(CA - expected_edge_length)]
    
    # 检查是否为有效等边三角形
    is_valid = all(error < tolerance for error in edge_errors)
    
    return is_valid, edge_lengths, edge_errors

def project_to_image(point_3d, id_value, cx=960, cy=540):
    """
    将3D点投影到图像平面（透视投影）
    
    参数：
        point_3d: 3D点坐标 [x, y, z]
        id_value: 像深值
        cx, cy: 主点坐标
    
    返回：
        image_point: 图像坐标 (u, v)
    """
    x, y, z = point_3d
    
    # 透视投影
    img_x = x * id_value / z + cx
    img_y = y * id_value / z + cy
    
    return (img_x, img_y)

def calc_image_depth(fov_h_deg, width_pixels):
    """
    从水平视角计算像深
    
    参数：
        fov_h_deg: 水平视角（度）
        width_pixels: 图像宽度（像素）
    
    返回：
        像深（像素）
    """
    fov_rad = np.deg2rad(fov_h_deg)
    return (width_pixels / 2) / np.tan(fov_rad / 2)

def backproject(A_img, B_img, C_img, id, cx=960, cy=540, edge_mm=50, max_iter=200):
    """
    反投影正三角形靶标，返回世界坐标（毫米）
    
    核心算法：通过迭代优化找到满足等边三角形约束的3D坐标
    
    参数：
        A_img, B_img, C_img: 三角形顶点的图像坐标 (u, v)
        id: 像深值（像素）
        cx, cy: 主点坐标（默认960, 540）
        edge_mm: 三角形边长（毫米，默认50）
        max_iter: 最大迭代次数（默认200）
    
    返回：
        A_3d, B_3d, C_3d: 三个顶点的3D世界坐标（毫米）
    """
    # 初始化三点坐标，z坐标都是id
    A_3d = np.array([A_img[0] - cx, A_img[1] - cy, id], dtype=float)
    B_3d = np.array([B_img[0] - cx, B_img[1] - cy, id], dtype=float)
    C_3d = np.array([C_img[0] - cx, C_img[1] - cy, id], dtype=float)
    
    # 计算射线方向（归一化）
    ray_B = np.array([B_img[0] - cx, B_img[1] - cy, id], dtype=float)
    ray_B = ray_B / np.linalg.norm(ray_B)
    
    ray_C = np.array([C_img[0] - cx, C_img[1] - cy, id], dtype=float)
    ray_C = ray_C / np.linalg.norm(ray_C)
    
    # 当前深度系数
    t_B = id
    t_C = id
    
    for iteration in range(max_iter):
        # 计算当前极差
        AB = np.linalg.norm(B_3d - A_3d)
        BC = np.linalg.norm(C_3d - B_3d)
        CA = np.linalg.norm(A_3d - C_3d)
        current_range = max(AB, BC, CA) - min(AB, BC, CA)
        
        # 收敛判断
        if current_range < 0.01:
            break
            
        # 极差作为步长
        step = current_range / 10.0
        
        # 步步为营：尝试B、C的伸缩
        best_range = current_range
        best_t_B = t_B
        best_t_C = t_C
        
        # 尝试9种组合
        for delta_B in [-1, 0, 1]:
            for delta_C in [-1, 0, 1]:
                # 新的深度
                trial_t_B = t_B + delta_B * step
                trial_t_C = t_C + delta_C * step
                
                if trial_t_B <= 0.1 or trial_t_C <= 0.1:
                    continue
                
                # 计算新位置
                trial_B = trial_t_B * ray_B
                trial_C = trial_t_C * ray_C
                
                # 计算试探的极差
                AB = np.linalg.norm(trial_B - A_3d)
                BC = np.linalg.norm(trial_C - trial_B)
                CA = np.linalg.norm(A_3d - trial_C)
                trial_range = max(AB, BC, CA) - min(AB, BC, CA)
                
                if trial_range < best_range:
                    best_range = trial_range
                    best_t_B = trial_t_B
                    best_t_C = trial_t_C
        
        # 更新位置
        t_B = best_t_B
        t_C = best_t_C
        B_3d = t_B * ray_B
        C_3d = t_C * ray_C
    
    # 计算平均边长（像素单位）
    AB = np.linalg.norm(B_3d - A_3d)
    BC = np.linalg.norm(C_3d - B_3d)
    CA = np.linalg.norm(A_3d - C_3d)
    avg_edge = (AB + BC + CA) / 3
    
    # 缩放到世界坐标（毫米）
    scale = edge_mm / avg_edge
    
    return A_3d * scale, B_3d * scale, C_3d * scale

def combine_F6(f6_a, f6_b):
    """
    F6变换的组合运算：f6_a * f6_b
    
    设计理念：F6→Q→计算→F6
    使用齐次矩阵进行计算，体现F6设计的核心思想
    
    参数：
        f6_a: 第一个F6变换 [tx, ty, tz, rx, ry, rz]，旋转单位为度
        f6_b: 第二个F6变换 [tx, ty, tz, rx, ry, rz]，旋转单位为度
    
    返回：
        f6_result: 组合后的F6变换
    """
    # F6→Q：转换为齐次矩阵进行计算
    Q_a = F62Q(f6_a)
    Q_b = F62Q(f6_b)
    
    # 矩阵乘法：Q_result = Q_a * Q_b
    Q_result = np.dot(Q_a, Q_b)
    
    # Q→F6：转换回F6表达
    f6_result = Q2F6(Q_result)
    
    return f6_result

def invF6(f6):
    """
    计算F6变换的逆变换
    
    设计理念：F6→Q→逆矩阵→F6
    使用齐次矩阵的逆矩阵计算，体现F6设计的核心思想
    
    参数：
        f6: F6变换 [tx, ty, tz, rx, ry, rz]，旋转单位为度
    
    返回：
        f6_inv: 逆F6变换
    """
    # F6→Q：转换为齐次矩阵
    Q = F62Q(f6)
    
    # 计算逆矩阵
    Q_inv = np.linalg.inv(Q)
    
    # Q→F6：转换回F6表达
    f6_inv = Q2F6(Q_inv)
    
    return f6_inv

def compute_relative_F6(f6_a, f6_b):
    """
    计算从相机A到相机B的相对F6变换
    
    相对变换 f6_AB = invF6(f6_A) * f6_B
    
    参数：
        f6_a: 相机A的F6变换（相机A在世界坐标系中的位姿）
        f6_b: 相机B的F6变换（相机B在世界坐标系中的位姿）
    
    返回：
        f6_ab: 从相机A到相机B的相对F6变换
    """
    f6_a_inv = invF6(f6_a)
    f6_ab = combine_F6(f6_a_inv, f6_b)
    return f6_ab

def F62Q(f6):
    """
    将F6变换转换为4×4齐次矩阵Q（用于计算）
    
    F6设计理念：F6是人类友好的表达方式，Q是计算友好的矩阵形式
    
    参数：
        f6: F6变换 [tx, ty, tz, rx, ry, rz]，旋转分量单位为度
    
    返回：
        Q: 4×4齐次变换矩阵
    """
    # 提取平移和旋转分量
    t = np.array(f6[0:3])
    r = np.array(f6[3:6])  # 度数单位
    
    # 将旋转向量转换为旋转矩阵
    R = o2r(r)
    
    # 构建4×4齐次矩阵
    Q = np.eye(4)
    Q[0:3, 0:3] = R
    Q[0:3, 3] = t
    
    return Q

def Q2F6(Q):
    """
    将4×4齐次矩阵Q转换回F6变换（用于表达）
    
    F6设计理念：从计算友好的矩阵形式回到人类友好的表达方式
    
    参数：
        Q: 4×4齐次变换矩阵
    
    返回：
        f6: F6变换 [tx, ty, tz, rx, ry, rz]，旋转分量单位为度
    """
    # 提取平移向量
    t = Q[0:3, 3]
    
    # 提取旋转矩阵
    R = Q[0:3, 0:3]
    
    # 将旋转矩阵转换为轴角表示（度数单位）
    r = r2o(R)
    
    # 组合成F6变换
    f6 = np.concatenate([t, r])
    
    return f6