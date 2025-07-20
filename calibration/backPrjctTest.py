#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('.')
from backPrjct import backproject_to_F6, CX, CY, EDGE_MM
from utils import calc_image_depth, backproject

def test_find_triangle():
    """测试find_triangle函数"""
    # 相机参数
    width = 1920
    fov = 60  # 度
    id = calc_image_depth(fov, width)
    print(f"像深: {id:.1f} 像素")
    print(f"主点: ({CX}, {CY})")
    
    # 模拟的图像坐标（一个大致的正三角形）
    A_img = (960, 500)   # 顶点
    B_img = (860, 620)   # 左下
    C_img = (1060, 620)  # 右下
    
    print(f"\n输入图像坐标:")
    print(f"A: {A_img}")
    print(f"B: {B_img}")
    print(f"C: {C_img}")
    
    # 找正三角形
    A_3d, B_3d, C_3d = backproject(A_img, B_img, C_img, id, CX, CY, EDGE_MM)
    
    print(f"\n找到的3D坐标（相机坐标系）:")
    print(f"A: [{A_3d[0]:.1f}, {A_3d[1]:.1f}, {A_3d[2]:.1f}]")
    print(f"B: [{B_3d[0]:.1f}, {B_3d[1]:.1f}, {B_3d[2]:.1f}]")
    print(f"C: [{C_3d[0]:.1f}, {C_3d[1]:.1f}, {C_3d[2]:.1f}]")
    
    # 计算边长
    AB = np.linalg.norm(B_3d - A_3d)
    BC = np.linalg.norm(C_3d - B_3d)
    CA = np.linalg.norm(A_3d - C_3d)
    
    print(f"\n边长:")
    print(f"AB: {AB:.3f}")
    print(f"BC: {BC:.3f}")
    print(f"CA: {CA:.3f}")
    print(f"极差: {max(AB,BC,CA) - min(AB,BC,CA):.3f}")
    
    # 测试反投影
    print(f"\n使用backproject函数（边长{EDGE_MM}mm）:")
    A_world, B_world, C_world = backproject(A_img, B_img, C_img, id, CX, CY, EDGE_MM)
    
    print(f"A: [{A_world[0]:.1f}, {A_world[1]:.1f}, {A_world[2]:.1f}] mm")
    print(f"B: [{B_world[0]:.1f}, {B_world[1]:.1f}, {B_world[2]:.1f}] mm")
    print(f"C: [{C_world[0]:.1f}, {C_world[1]:.1f}, {C_world[2]:.1f}] mm")
    
    # 验证世界坐标下的边长
    AB_w = np.linalg.norm(B_world - A_world)
    BC_w = np.linalg.norm(C_world - B_world)
    CA_w = np.linalg.norm(A_world - C_world)
    
    print(f"\n世界坐标边长验证:")
    print(f"AB: {AB_w:.1f} mm")
    print(f"BC: {BC_w:.1f} mm")
    print(f"CA: {CA_w:.1f} mm")
    print(f"期望: {EDGE_MM} mm")

def test_backproject_to_F6():
    """测试一体化F6生成功能"""
    print(f"\n=== 测试一体化F6生成 ===")
    
    # 相机参数
    width = 1920
    fov = 60  # 度
    id = calc_image_depth(fov, width)
    print(f"像深: {id:.1f} 像素")
    
    # 模拟的图像坐标
    A_img = (960, 500)   # 顶点
    B_img = (860, 620)   # 左下
    C_img = (1060, 620)  # 右下
    
    print(f"输入图像坐标:")
    print(f"A: {A_img}")
    print(f"B: {B_img}")
    print(f"C: {C_img}")
    
    # 一体化生成F6
    f6 = backproject_to_F6(A_img, B_img, C_img, id)
    
    print(f"\n一体化生成的F6参数:")
    print(f"平移: [{f6[0]:.1f}, {f6[1]:.1f}, {f6[2]:.1f}] mm")
    print(f"旋转: [{f6[3]:.3f}, {f6[4]:.3f}, {f6[5]:.3f}] rad")
    
    # 计算旋转角度
    rotation_angle = np.linalg.norm(f6[3:6])
    print(f"旋转角度: {np.rad2deg(rotation_angle):.1f}°")
    
    # 对比两步法的结果
    print(f"\n=== 对比验证 ===")
    A_3d, B_3d, C_3d = backproject(A_img, B_img, C_img, id, CX, CY, EDGE_MM)
    print(f"两步法的3D坐标:")
    print(f"A: [{A_3d[0]:.1f}, {A_3d[1]:.1f}, {A_3d[2]:.1f}] mm")
    print(f"B: [{B_3d[0]:.1f}, {B_3d[1]:.1f}, {B_3d[2]:.1f}] mm")
    print(f"C: [{C_3d[0]:.1f}, {C_3d[1]:.1f}, {C_3d[2]:.1f}] mm")
    
    # F6平移应该等于A点坐标
    print(f"F6平移与A点对比:")
    print(f"F6平移: [{f6[0]:.1f}, {f6[1]:.1f}, {f6[2]:.1f}]")
    print(f"A点坐标: [{A_3d[0]:.1f}, {A_3d[1]:.1f}, {A_3d[2]:.1f}]")
    print(f"差异: {np.linalg.norm(f6[0:3] - A_3d):.3f} mm")

if __name__ == "__main__":
    test_find_triangle()
    test_backproject_to_F6()