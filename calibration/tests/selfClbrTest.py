#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from backPrjct import CX, CY, EDGE_MM, MAX_ITER
from utils import calc_image_depth, backproject
from selfClbr import SelfCalibration

def create_accurate_test_data():
    """创建准确的测试数据"""
    
    # 真实参数
    true_fov = 58.0  # 真实FOV
    true_id = calc_image_depth(true_fov, 1920)
    
    print(f"真实FOV: {true_fov}°")
    print(f"真实像深: {true_id:.1f} 像素")
    
    # 基础三角形（在图像中心附近）
    base_A = (960, 450)
    base_B = (860, 570)
    base_C = (1060, 570)
    
    # 第一个位置：基础位置
    obs1 = (base_A, base_B, base_C)
    
    # 计算第一个位置的3D坐标
    A1_3d, B1_3d, C1_3d = backproject(base_A, base_B, base_C, true_id, CX, CY, EDGE_MM, MAX_ITER)
    center1 = (A1_3d + B1_3d + C1_3d) / 3
    
    print(f"位置1中心: [{center1[0]:.1f}, {center1[1]:.1f}, {center1[2]:.1f}] mm")
    
    # 定义实际平移（必须包含纵深成分）
    actual_translation = np.array([30.0, 20.0, 100.0])  # 向右30mm, 向上20mm, 向前100mm
    
    # 计算第二个位置的3D坐标
    A2_3d = A1_3d + actual_translation
    B2_3d = B1_3d + actual_translation
    C2_3d = C1_3d + actual_translation
    
    # 将3D坐标投影回图像平面
    A2_img = project_to_image(A2_3d, true_id)
    B2_img = project_to_image(B2_3d, true_id)
    C2_img = project_to_image(C2_3d, true_id)
    
    obs2 = (A2_img, B2_img, C2_img)
    
    print(f"位置1图像: A{obs1[0]}, B{obs1[1]}, C{obs1[2]}")
    print(f"位置2图像: A{obs2[0]}, B{obs2[1]}, C{obs2[2]}")
    print(f"实际平移: [{actual_translation[0]:.1f}, {actual_translation[1]:.1f}, {actual_translation[2]:.1f}] mm")
    
    return [(obs1, obs2, actual_translation)]

def project_to_image(point_3d, id_value):
    """将3D点投影到图像平面"""
    x, y, z = point_3d
    
    # 透视投影
    img_x = x * id_value / z + CX
    img_y = y * id_value / z + CY
    
    return (img_x, img_y)

def test_accurate_calibration():
    """测试准确的自标定"""
    print("=== 准确自标定测试 ===")
    
    # 创建测试数据
    observation_pairs = create_accurate_test_data()
    
    # 模拟厂家FOV（有偏差）
    manufacturer_fov = 60.0
    
    # 创建自标定对象
    calibrator = SelfCalibration()
    
    # 执行自标定
    optimal_id, optimal_fov, error = calibrator.binary_search_calibration(
        observation_pairs, 
        manufacturer_fov, 
        search_range_percent=10,
        tolerance=0.01
    )
    
    # 分析结果
    errors = calibrator.analyze_translation_accuracy(optimal_id, observation_pairs)
    
    # 与真实值比较
    true_fov = 58.0
    true_id = calc_image_depth(true_fov, 1920)
    
    print(f"\n=== 与真实值比较 ===")
    print(f"真实FOV: {true_fov}°")
    print(f"标定FOV: {optimal_fov:.2f}°")
    print(f"FOV误差: {abs(optimal_fov - true_fov):.2f}°")
    print(f"真实像深: {true_id:.2f} 像素")
    print(f"标定像深: {optimal_id:.2f} 像素")
    print(f"像深误差: {abs(optimal_id - true_id):.2f} 像素")

if __name__ == "__main__":
    test_accurate_calibration()