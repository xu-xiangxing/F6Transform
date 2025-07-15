#!/usr/bin/env python3

import numpy as np
from utils import r2o, backproject, tri2F6

# 系统参数
CX = 960  # 主点x坐标
CY = 540  # 主点y坐标

# 宏变量
MAX_ITER = 200  # 最大迭代次数
EDGE_MM = 50    # 靶标正三角形边长（毫米）

def backproject_to_F6(A_img, B_img, C_img, id):
    """
    反投影正三角形靶标，直接返回F6变换参数
    
    从图像坐标一步到位生成靶标坐标系的F6参数
    靶标坐标系定义：
    - 原点：A顶点
    - X轴：AB方向
    - Z轴：AB × AC方向（右手法则）
    - Y轴：Z × X方向
    
    参数：
        A_img, B_img, C_img: 三角形顶点图像坐标
        id: 像深值
        
    返回：
        f6: [tx, ty, tz, rx, ry, rz] - F6变换参数
    """
    # 先反投影得到3D坐标，使用默认参数
    A_3d, B_3d, C_3d = backproject(A_img, B_img, C_img, id, CX, CY, EDGE_MM, MAX_ITER)
    
    # 使用utils中的tri2F6函数构建F6参数
    f6 = tri2F6(A_3d, B_3d, C_3d)
    
    return f6

if __name__ == "__main__":
    # 演示反投影功能
    from utils import calc_image_depth
    
    print("=== 反投影算法演示 ===")
    
    # 相机参数
    width = 1920
    fov = 60  # 度
    id = calc_image_depth(fov, width)
    print(f"像深: {id:.1f} 像素")
    print(f"主点: ({CX}, {CY})")
    
    # 模拟的图像坐标（一个大致的正三角形，相对中心偏移40mm）
    # 40mm偏移对应的像素偏移 = 40 * id / 假设深度
    offset_pixels = 40 * id / 1000  # 假设1000mm深度
    
    A_img = (960 + offset_pixels, 500 - offset_pixels)   # 顶点：向右上偏移
    B_img = (860 + offset_pixels, 620 - offset_pixels)   # 左下：向右上偏移
    C_img = (1060 + offset_pixels, 620 - offset_pixels)  # 右下：向右上偏移
    
    print(f"\n输入图像坐标（相对中心偏移约40mm）:")
    print(f"A: ({A_img[0]:.1f}, {A_img[1]:.1f})")
    print(f"B: ({B_img[0]:.1f}, {B_img[1]:.1f})")
    print(f"C: ({C_img[0]:.1f}, {C_img[1]:.1f})")
    
    # 反投影得到3D坐标
    A_3d, B_3d, C_3d = backproject(A_img, B_img, C_img, id, CX, CY, EDGE_MM, MAX_ITER)
    
    print(f"\n反投影得到的3D坐标（世界坐标系）:")
    print(f"A: [{A_3d[0]:.1f}, {A_3d[1]:.1f}, {A_3d[2]:.1f}] mm")
    print(f"B: [{B_3d[0]:.1f}, {B_3d[1]:.1f}, {B_3d[2]:.1f}] mm")
    print(f"C: [{C_3d[0]:.1f}, {C_3d[1]:.1f}, {C_3d[2]:.1f}] mm")
    
    # 计算三角形中心相对原点的位置
    center = (A_3d + B_3d + C_3d) / 3
    center_distance = np.linalg.norm(center)
    print(f"\n三角形中心: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] mm")
    print(f"中心到原点距离: {center_distance:.1f} mm（应该接近40mm的投影）")
    
    # 验证边长
    AB = np.linalg.norm(B_3d - A_3d)
    BC = np.linalg.norm(C_3d - B_3d)
    CA = np.linalg.norm(A_3d - C_3d)
    
    print(f"\n边长验证:")
    print(f"AB: {AB:.1f} mm")
    print(f"BC: {BC:.1f} mm")
    print(f"CA: {CA:.1f} mm")
    print(f"期望: {EDGE_MM} mm")
    
    # 生成F6参数
    f6 = backproject_to_F6(A_img, B_img, C_img, id)
    
    print(f"\n生成的F6参数:")
    print(f"平移: [{f6[0]:.1f}, {f6[1]:.1f}, {f6[2]:.1f}] mm")
    print(f"旋转: [{f6[3]:.3f}, {f6[4]:.3f}, {f6[5]:.3f}] rad")
    
    # 计算旋转角度
    rotation_angle = np.linalg.norm(f6[3:6])
    print(f"旋转角度: {np.rad2deg(rotation_angle):.1f}°")

