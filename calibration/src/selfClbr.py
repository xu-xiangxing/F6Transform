#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils import calc_image_depth, backproject

# 系统参数
CX = 960  # 主点x坐标
CY = 540  # 主点y坐标
EDGE_MM = 50  # 靶标正三角形边长（毫米）
MAX_ITER = 200  # 最大迭代次数

class SelfCalibration:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        
    def calc_translation_error(self, id_value, observation_pairs):
        """
        计算平移距离误差
        observation_pairs: [(obs1, obs2, actual_translation), ...]
        obs1, obs2: (A_img, B_img, C_img)
        actual_translation: 实际平移距离 (dx, dy, dz) mm
        """
        total_error = 0
        valid_pairs = 0
        
        for obs1, obs2, actual_translation in observation_pairs:
            try:
                # 反投影第一个位置的三角形
                A1_3d, B1_3d, C1_3d = backproject(obs1[0], obs1[1], obs1[2], id_value, CX, CY, EDGE_MM, MAX_ITER)
                
                # 反投影第二个位置的三角形
                A2_3d, B2_3d, C2_3d = backproject(obs2[0], obs2[1], obs2[2], id_value, CX, CY, EDGE_MM, MAX_ITER)
                
                # 计算三角形中心的平移
                center1 = (A1_3d + B1_3d + C1_3d) / 3
                center2 = (A2_3d + B2_3d + C2_3d) / 3
                
                # 计算的平移距离
                calculated_translation = center2 - center1
                
                # 与实际平移距离的误差
                actual_trans = np.array(actual_translation)
                translation_error = np.linalg.norm(calculated_translation - actual_trans)
                
                total_error += translation_error
                valid_pairs += 1
                
            except Exception as e:
                continue
        
        if valid_pairs == 0:
            return float('inf')
        
        return total_error / valid_pairs
    
    def binary_search_calibration(self, observation_pairs, manufacturer_fov, search_range_percent=10, tolerance=0.01):
        """
        基于二分法的自标定算法
        manufacturer_fov: 厂家提供的FOV（度）
        search_range_percent: 搜索范围百分比
        tolerance: 收敛阈值
        """
        # 从厂家FOV计算初始像深
        m = calc_image_depth(manufacturer_fov, self.width)
        
        # 计算搜索范围 [a, b]
        range_delta = m * search_range_percent / 100
        a = m - range_delta
        b = m + range_delta
        
        print(f"自标定算法启动...")
        print(f"厂家FOV: {manufacturer_fov}°")
        print(f"初始像深: {m:.2f} 像素")
        print(f"搜索范围: [{a:.2f}, {b:.2f}] 像素")
        print(f"观测对数量: {len(observation_pairs)}")
        
        # 二分法搜索
        iteration = 0
        while (b - a) > tolerance:
            iteration += 1
            
            # 计算中点
            mid = (a + b) / 2
            
            # 计算左右两个测试点的误差
            left_point = a + (mid - a) * 0.618  # 黄金分割点
            right_point = mid + (b - mid) * 0.618
            
            error_left = self.calc_translation_error(left_point, observation_pairs)
            error_right = self.calc_translation_error(right_point, observation_pairs)
            
            print(f"迭代 {iteration}: 范围[{a:.3f}, {b:.3f}], 误差 left={error_left:.3f}, right={error_right:.3f}")
            
            # 根据误差大小调整搜索范围
            if error_left < error_right:
                b = right_point
            else:
                a = left_point
        
        # 最终结果
        optimal_id = (a + b) / 2
        final_error = self.calc_translation_error(optimal_id, observation_pairs)
        
        # 转换回FOV
        optimal_fov = np.rad2deg(2 * np.arctan(self.width / (2 * optimal_id)))
        
        print(f"\n自标定结果:")
        print(f"收敛迭代次数: {iteration}")
        print(f"最优像深: {optimal_id:.3f} 像素")
        print(f"对应FOV: {optimal_fov:.3f}°")
        print(f"与厂家FOV差异: {abs(optimal_fov - manufacturer_fov):.3f}°")
        print(f"平移距离误差: {final_error:.3f} mm")
        
        return optimal_id, optimal_fov, final_error
    
    def analyze_translation_accuracy(self, id_value, observation_pairs):
        """
        分析平移精度
        """
        print(f"\n=== 平移精度分析 ===")
        print(f"使用像深: {id_value:.3f} 像素")
        
        errors = []
        
        for i, (obs1, obs2, actual_translation) in enumerate(observation_pairs):
            try:
                # 反投影计算
                A1_3d, B1_3d, C1_3d = backproject(obs1[0], obs1[1], obs1[2], id_value, CX, CY, EDGE_MM, MAX_ITER)
                A2_3d, B2_3d, C2_3d = backproject(obs2[0], obs2[1], obs2[2], id_value, CX, CY, EDGE_MM, MAX_ITER)
                
                # 计算中心平移
                center1 = (A1_3d + B1_3d + C1_3d) / 3
                center2 = (A2_3d + B2_3d + C2_3d) / 3
                calculated_translation = center2 - center1
                
                # 误差分析
                actual_trans = np.array(actual_translation)
                translation_error = np.linalg.norm(calculated_translation - actual_trans)
                
                errors.append(translation_error)
                
                print(f"观测对{i+1}:")
                print(f"  实际平移: [{actual_trans[0]:.1f}, {actual_trans[1]:.1f}, {actual_trans[2]:.1f}] mm")
                print(f"  计算平移: [{calculated_translation[0]:.1f}, {calculated_translation[1]:.1f}, {calculated_translation[2]:.1f}] mm")
                print(f"  误差: {translation_error:.2f} mm")
                
            except Exception as e:
                print(f"观测对{i+1}: 处理失败 - {e}")
        
        if errors:
            print(f"\n统计结果:")
            print(f"平均平移误差: {np.mean(errors):.2f} mm")
            print(f"标准差: {np.std(errors):.2f} mm")
            print(f"最大误差: {np.max(errors):.2f} mm")
            print(f"最小误差: {np.min(errors):.2f} mm")
            
            # 判断标定质量
            if np.mean(errors) < 5.0:
                print("✓ 自标定质量: 优秀")
            elif np.mean(errors) < 10.0:
                print("△ 自标定质量: 良好")
            else:
                print("✗ 自标定质量: 需要改进")
        
        return errors

def create_translation_test_data():
    """
    创建平移测试数据
    """
    # 假设真实FOV为58度
    true_fov = 58
    true_id = calc_image_depth(true_fov, 1920)
    
    observation_pairs = []
    
    # 模拟几组平移测试
    base_center = (960, 540)
    base_radius = 80
    
    # 定义几个平移向量 (dx, dy, dz) - 包含纵深成分
    translations = [
        (100, 0, 0),      # 纯X平移
        (0, 100, 0),      # 纯Y平移
        (0, 0, 200),      # 纯Z平移（纵深）
        (50, 50, 100),    # 复合平移
        (-80, 30, -150),  # 复合平移
        (0, -60, 300),    # Y+Z平移
        (120, 0, -100)    # X+Z平移
    ]
    
    for dx, dy, dz in translations:
        # 第一个位置的三角形
        A1_img = (base_center[0], base_center[1] - base_radius)
        B1_img = (base_center[0] - base_radius * 0.866, base_center[1] + base_radius * 0.5)
        C1_img = (base_center[0] + base_radius * 0.866, base_center[1] + base_radius * 0.5)
        
        # 模拟平移后的图像位置变化
        # 考虑透视变换
        base_depth = 1000  # 基准深度
        new_depth = base_depth + dz
        scale_factor = base_depth / new_depth
        
        # 第二个位置的三角形（考虑透视投影）
        proj_dx = dx * true_id / base_depth
        proj_dy = dy * true_id / base_depth
        new_radius = base_radius * scale_factor
        
        A2_img = (base_center[0] + proj_dx, base_center[1] + proj_dy - new_radius)
        B2_img = (base_center[0] + proj_dx - new_radius * 0.866, base_center[1] + proj_dy + new_radius * 0.5)
        C2_img = (base_center[0] + proj_dx + new_radius * 0.866, base_center[1] + proj_dy + new_radius * 0.5)
        
        # 添加一些噪声模拟真实检测误差
        noise_level = 0.5
        A2_img = (A2_img[0] + np.random.normal(0, noise_level), A2_img[1] + np.random.normal(0, noise_level))
        B2_img = (B2_img[0] + np.random.normal(0, noise_level), B2_img[1] + np.random.normal(0, noise_level))
        C2_img = (C2_img[0] + np.random.normal(0, noise_level), C2_img[1] + np.random.normal(0, noise_level))
        
        obs1 = (A1_img, B1_img, C1_img)
        obs2 = (A2_img, B2_img, C2_img)
        actual_translation = (dx, dy, dz)
        
        observation_pairs.append((obs1, obs2, actual_translation))
    
    return observation_pairs

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
    # 创建自标定对象
    calibrator = SelfCalibration()
    
    # 生成测试数据
    observation_pairs = create_translation_test_data()
    
    # 模拟厂家提供的FOV（故意设置一个略有偏差的值）
    manufacturer_fov = 60.0  # 真实值是58度
    
    # 执行自标定
    optimal_id, optimal_fov, error = calibrator.binary_search_calibration(
        observation_pairs, 
        manufacturer_fov, 
        search_range_percent=10,
        tolerance=0.01
    )
    
    # 分析标定质量
    errors = calibrator.analyze_translation_accuracy(optimal_id, observation_pairs)
    
    # 运行准确测试
    test_accurate_calibration()