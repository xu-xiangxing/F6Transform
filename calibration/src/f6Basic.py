#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from backPrjct import backproject_to_F6, CX, CY, EDGE_MM
from utils import calc_image_depth, backproject, combine_F6, invF6, compute_relative_F6, apply_F6_transform

def test_step1_basic_projection_consistency():
    """步骤1：验证最基本的反投影-投影一致性"""
    print("=== 步骤1：基本投影一致性测试 ===")
    
    # 基础参数
    width = 1920
    fov = 60.0
    id_value = calc_image_depth(fov, width)
    print(f"FOV: {fov}°, 像深: {id_value:.1f} 像素")
    
    # 定义一个简单的图像坐标点
    img_point = (1000.0, 600.0)
    print(f"原始图像坐标: {img_point}")
    
    # 反投影到z=id平面的3D点
    x_3d = img_point[0] - CX
    y_3d = img_point[1] - CY
    z_3d = id_value
    point_3d = np.array([x_3d, y_3d, z_3d])
    print(f"反投影3D坐标: [{point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}]")
    
    # 重新投影回图像坐标
    u_reproject = point_3d[0] * id_value / point_3d[2] + CX
    v_reproject = point_3d[1] * id_value / point_3d[2] + CY
    reproject_point = (u_reproject, v_reproject)
    print(f"重投影图像坐标: ({reproject_point[0]:.1f}, {reproject_point[1]:.1f})")
    
    # 验证一致性
    error = np.sqrt((img_point[0] - reproject_point[0])**2 + (img_point[1] - reproject_point[1])**2)
    print(f"投影误差: {error:.10f} 像素")
    
    success = error < 1e-10
    print(f"步骤1结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step2_f6_transform_identity():
    """步骤2：验证F6变换的单位元性质"""
    print("\n=== 步骤2：F6变换单位元测试 ===")
    
    # 测试点
    test_point = np.array([10.0, 20.0, 300.0])
    print(f"测试点: [{test_point[0]:.1f}, {test_point[1]:.1f}, {test_point[2]:.1f}]")
    
    # 单位F6变换（零变换）
    f6_identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"单位F6: [{f6_identity[0]:.1f}, {f6_identity[1]:.1f}, {f6_identity[2]:.1f}, {f6_identity[3]:.1f}, {f6_identity[4]:.1f}, {f6_identity[5]:.1f}]")
    
    # 应用F6变换
    transformed_point = apply_F6_transform(test_point, f6_identity)
    print(f"变换后: [{transformed_point[0]:.1f}, {transformed_point[1]:.1f}, {transformed_point[2]:.1f}]")
    
    # 验证单位元性质
    error = np.linalg.norm(transformed_point - test_point)
    print(f"单位元误差: {error:.10f}")
    
    success = error < 1e-10
    print(f"步骤2结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step3_f6_inverse():
    """步骤3：验证F6逆变换"""
    print("\n=== 步骤3：F6逆变换测试 ===")
    
    # 测试点和F6变换
    test_point = np.array([10.0, 20.0, 300.0])
    f6_test = np.array([5.0, -3.0, 10.0, 0.0, 0.0, 0.0])  # 只测试平移
    print(f"测试点: [{test_point[0]:.1f}, {test_point[1]:.1f}, {test_point[2]:.1f}]")
    print(f"测试F6: [{f6_test[0]:.1f}, {f6_test[1]:.1f}, {f6_test[2]:.1f}, {f6_test[3]:.1f}, {f6_test[4]:.1f}, {f6_test[5]:.1f}]")
    
    # 正向变换
    forward = apply_F6_transform(test_point, f6_test)
    print(f"正向变换: [{forward[0]:.1f}, {forward[1]:.1f}, {forward[2]:.1f}]")
    
    # 逆变换
    f6_inv = invF6(f6_test)
    backward = apply_F6_transform(forward, f6_inv)
    print(f"逆变换: [{backward[0]:.1f}, {backward[1]:.1f}, {backward[2]:.1f}]")
    
    # 验证
    error = np.linalg.norm(backward - test_point)
    print(f"逆变换误差: {error:.10f}")
    
    success = error < 1e-10
    print(f"步骤3结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step4_f6_combine():
    """步骤4：验证F6群组合运算"""
    print("\n=== 步骤4：F6群组合测试 ===")
    
    # 两个简单的F6变换（只包含平移）
    f6_1 = np.array([10.0, 5.0, 2.0, 0.0, 0.0, 0.0])
    f6_2 = np.array([3.0, -2.0, 8.0, 0.0, 0.0, 0.0])
    print(f"F6_1: [{f6_1[0]:.1f}, {f6_1[1]:.1f}, {f6_1[2]:.1f}, {f6_1[3]:.1f}, {f6_1[4]:.1f}, {f6_1[5]:.1f}]")
    print(f"F6_2: [{f6_2[0]:.1f}, {f6_2[1]:.1f}, {f6_2[2]:.1f}, {f6_2[3]:.1f}, {f6_2[4]:.1f}, {f6_2[5]:.1f}]")
    
    # 测试点
    test_point = np.array([0.0, 0.0, 100.0])
    print(f"测试点: [{test_point[0]:.1f}, {test_point[1]:.1f}, {test_point[2]:.1f}]")
    
    # 方法1：分步变换
    step1 = apply_F6_transform(test_point, f6_1)
    step2 = apply_F6_transform(step1, f6_2)
    print(f"分步变换结果: [{step2[0]:.1f}, {step2[1]:.1f}, {step2[2]:.1f}]")
    
    # 方法2：组合后变换
    f6_combined = combine_F6(f6_1, f6_2)
    combined_result = apply_F6_transform(test_point, f6_combined)
    print(f"组合F6: [{f6_combined[0]:.1f}, {f6_combined[1]:.1f}, {f6_combined[2]:.1f}, {f6_combined[3]:.1f}, {f6_combined[4]:.1f}, {f6_combined[5]:.1f}]")
    print(f"组合变换结果: [{combined_result[0]:.1f}, {combined_result[1]:.1f}, {combined_result[2]:.1f}]")
    
    # 验证
    error = np.linalg.norm(step2 - combined_result)
    print(f"组合运算误差: {error:.10f}")
    
    success = error < 1e-10
    print(f"步骤4结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step5_backproject_consistency():
    """步骤5：验证backproject函数的一致性"""
    print("\n=== 步骤5：backproject一致性测试 ===")
    
    # 基础参数
    width = 1920
    fov = 60.0
    id_value = calc_image_depth(fov, width)
    
    # 简单的三角形图像坐标
    A_img = (960.0, 700.0)
    B_img = (900.0, 600.0)  
    C_img = (1020.0, 600.0)
    print(f"图像坐标 A: {A_img}, B: {B_img}, C: {C_img}")
    
    # 反投影得到3D坐标
    triangle_3d = backproject(A_img, B_img, C_img, id_value)
    print(f"反投影3D坐标:")
    print(f"A: [{triangle_3d[0][0]:.1f}, {triangle_3d[0][1]:.1f}, {triangle_3d[0][2]:.1f}]")
    print(f"B: [{triangle_3d[1][0]:.1f}, {triangle_3d[1][1]:.1f}, {triangle_3d[1][2]:.1f}]")
    print(f"C: [{triangle_3d[2][0]:.1f}, {triangle_3d[2][1]:.1f}, {triangle_3d[2][2]:.1f}]")
    
    # 重新投影
    A_reproj = (triangle_3d[0][0] * id_value / triangle_3d[0][2] + CX,
                triangle_3d[0][1] * id_value / triangle_3d[0][2] + CY)
    B_reproj = (triangle_3d[1][0] * id_value / triangle_3d[1][2] + CX,
                triangle_3d[1][1] * id_value / triangle_3d[1][2] + CY)
    C_reproj = (triangle_3d[2][0] * id_value / triangle_3d[2][2] + CX,
                triangle_3d[2][1] * id_value / triangle_3d[2][2] + CY)
    
    print(f"重投影坐标 A: ({A_reproj[0]:.1f}, {A_reproj[1]:.1f})")
    print(f"重投影坐标 B: ({B_reproj[0]:.1f}, {B_reproj[1]:.1f})")
    print(f"重投影坐标 C: ({C_reproj[0]:.1f}, {C_reproj[1]:.1f})")
    
    # 计算误差
    error_A = np.sqrt((A_img[0] - A_reproj[0])**2 + (A_img[1] - A_reproj[1])**2)
    error_B = np.sqrt((B_img[0] - B_reproj[0])**2 + (B_img[1] - B_reproj[1])**2)
    error_C = np.sqrt((C_img[0] - C_reproj[0])**2 + (C_img[1] - C_reproj[1])**2)
    
    max_error = max(error_A, error_B, error_C)
    print(f"重投影误差 A: {error_A:.10f}, B: {error_B:.10f}, C: {error_C:.10f}")
    print(f"最大误差: {max_error:.10f}")
    
    success = max_error < 1e-6  # backproject函数可能有迭代误差
    print(f"步骤5结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step6_f6_group_properties():
    """步骤6：验证F6群的数学性质"""
    print("\n=== 步骤6：F6群运算性质测试 ===")
    
    # 定义两个简单的F6变换
    f6_1 = np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0])  # 纯平移
    f6_2 = np.array([5.0, -10.0, 15.0, 0.0, 0.0, 0.0])  # 纯平移
    
    print(f"F6_1: [{f6_1[0]:.1f}, {f6_1[1]:.1f}, {f6_1[2]:.1f}, {f6_1[3]:.1f}, {f6_1[4]:.1f}, {f6_1[5]:.1f}]")
    print(f"F6_2: [{f6_2[0]:.1f}, {f6_2[1]:.1f}, {f6_2[2]:.1f}, {f6_2[3]:.1f}, {f6_2[4]:.1f}, {f6_2[5]:.1f}]")
    
    # 测试点
    test_point = np.array([1.0, 2.0, 3.0])
    print(f"测试点: [{test_point[0]:.1f}, {test_point[1]:.1f}, {test_point[2]:.1f}]")
    
    # 验证结合律：(F6_1 ⊕ F6_2) ⊕ point = F6_1 ⊕ (F6_2 ⊕ point)
    step1 = apply_F6_transform(test_point, f6_1)
    step2 = apply_F6_transform(step1, f6_2)
    
    f6_combined = combine_F6(f6_1, f6_2)
    combined_result = apply_F6_transform(test_point, f6_combined)
    
    associative_error = np.linalg.norm(step2 - combined_result)
    print(f"结合律验证误差: {associative_error:.10f}")
    
    # 验证逆元性质：F6_1 ⊕ !F6_1 = E
    f6_1_inv = invF6(f6_1)
    identity = combine_F6(f6_1, f6_1_inv)
    identity_error = np.linalg.norm(identity)
    print(f"逆元验证误差: {identity_error:.10f}")
    
    # 验证群方程求解：F6_1 ⊕ F6_X = F6_combined，求F6_X
    f6_X_solved = combine_F6(f6_1_inv, f6_combined)
    solve_error = np.linalg.norm(f6_X_solved - f6_2)
    print(f"方程求解误差: {solve_error:.10f}")
    
    success = (associative_error < 1e-10) and (identity_error < 1e-10) and (solve_error < 1e-10)
    print(f"步骤6结果: {'✓ 通过' if success else '✗ 失败'}")
    return success

def test_step7_f6_meaning():
    """步骤7：探索F6参数的物理意义"""
    print("\n=== 步骤7：F6参数意义探索测试 ===")
    
    # 基础参数
    width = 1920
    fov = 60.0
    id_value = calc_image_depth(fov, width)
    
    # 简单的图像坐标
    A_img = (960.0, 700.0)
    B_img = (900.0, 600.0)
    C_img = (1020.0, 600.0)
    print(f"图像坐标: A{A_img}, B{B_img}, C{C_img}")
    
    # 反投影得到3D坐标
    triangle_3d = backproject(A_img, B_img, C_img, id_value)
    world_A = np.array(triangle_3d[0])
    
    # 计算F6参数
    f6 = backproject_to_F6(A_img, B_img, C_img, id_value)
    print(f"F6参数: [{f6[0]:.1f}, {f6[1]:.1f}, {f6[2]:.1f}, {f6[3]:.1f}, {f6[4]:.1f}, {f6[5]:.1f}]")
    
    # 关键验证：F6前三个参数与A点坐标的关系
    print(f"F6前三个参数: [{f6[0]:.1f}, {f6[1]:.1f}, {f6[2]:.1f}]")
    print(f"A点3D坐标:    [{world_A[0]:.1f}, {world_A[1]:.1f}, {world_A[2]:.1f}]")
    coord_match = np.allclose(f6[:3], world_A, atol=1e-6)
    print(f"F6平移分量 = A点坐标: {coord_match}")
    
    # 测试零点变换理解F6几何意义
    zero_point = np.array([0.0, 0.0, 0.0])
    transformed_zero = apply_F6_transform(zero_point, f6)
    zero_to_A_match = np.allclose(transformed_zero, world_A, atol=1e-6)
    print(f"F6变换零点 = A点坐标: {zero_to_A_match}")
    
    # 测试F6逆变换的意义
    f6_inv = invF6(f6)
    inv_transformed_A = apply_F6_transform(world_A, f6_inv)
    A_to_zero_match = np.allclose(inv_transformed_A, zero_point, atol=1e-6)
    print(f"F6逆变换A点 = 零点: {A_to_zero_match}")
    
    success = coord_match and zero_to_A_match and A_to_zero_match
    print(f"步骤7结果: {'✓ 通过' if success else '✗ 失败'}")
    if success:
        print("✓ F6的物理意义：将靶标坐标系原点变换到相机坐标系中A点的位置")
    return success

if __name__ == "__main__":
    print("F6Transform 逐步验证测试")
    print("=" * 50)
    
    results = []
    results.append(test_step1_basic_projection_consistency())
    results.append(test_step2_f6_transform_identity())
    results.append(test_step3_f6_inverse())
    results.append(test_step4_f6_combine())
    results.append(test_step5_backproject_consistency())
    results.append(test_step6_f6_group_properties())
    results.append(test_step7_f6_meaning())
    
    print("\n" + "=" * 50)
    print("总结:")
    step_names = ["基本投影一致性", "F6单位元", "F6逆变换", "F6群组合", "backproject一致性", "F6群运算性质", "F6参数意义"]
    for i, (name, result) in enumerate(zip(step_names, results)):
        print(f"步骤{i+1} {name}: {'✓ 通过' if result else '✗ 失败'}")
    
    all_passed = all(results)
    print(f"\n整体结果: {'🎉 所有步骤通过' if all_passed else '❌ 存在失败步骤'}")