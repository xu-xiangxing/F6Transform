#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from backPrjct import backproject_to_F6, CX, CY, EDGE_MM
from utils import calc_image_depth, backproject, apply_F6_transform

def correct_dual_camera_test():
    """用正确的直接坐标变换方法进行双相机测试"""
    print("=== 正确的双相机标定测试 ===")
    
    # 基础参数
    width = 1920
    fov_A = 60.0
    fov_B = 58.0
    id_A = calc_image_depth(fov_A, width)
    id_B = calc_image_depth(fov_B, width)
    print(f"相机A - FOV: {fov_A}°, 像深: {id_A:.1f} 像素")
    print(f"相机B - FOV: {fov_B}°, 像深: {id_B:.1f} 像素")
    
    # 步骤1：相机A的图像坐标（已知观测）
    A_img_A = (960.0, 700.0)
    B_img_A = (900.0, 600.0)
    C_img_A = (1020.0, 600.0)
    print(f"\n步骤1：相机A观测到的三角形")
    print(f"A: {A_img_A}, B: {B_img_A}, C: {C_img_A}")
    
    # 步骤2：给定相机间相对变换F6_AB
    f6_AB = np.array([10.0, 5.0, 2.0, 0.0, 0.0, 0.0])  # 已知的相对变换
    print(f"\n步骤2：给定相机间相对变换F6_AB")
    print(f"F6_AB: [{f6_AB[0]:.1f}, {f6_AB[1]:.1f}, {f6_AB[2]:.1f}, {f6_AB[3]:.1f}, {f6_AB[4]:.1f}, {f6_AB[5]:.1f}]")
    
    # 步骤3：从相机A反投影得到3D坐标
    triangle_3d_A = backproject(A_img_A, B_img_A, C_img_A, id_A)
    world_A = np.array(triangle_3d_A[0])
    world_B = np.array(triangle_3d_A[1])
    world_C = np.array(triangle_3d_A[2])
    print(f"\n步骤3：从相机A反投影得到3D坐标")
    print(f"A: [{world_A[0]:.1f}, {world_A[1]:.1f}, {world_A[2]:.1f}]")
    print(f"B: [{world_B[0]:.1f}, {world_B[1]:.1f}, {world_B[2]:.1f}]")
    print(f"C: [{world_C[0]:.1f}, {world_C[1]:.1f}, {world_C[2]:.1f}]")
    
    # 步骤4：用F6_AB将3D坐标从相机A坐标系变换到相机B坐标系
    world_A_in_B = apply_F6_transform(world_A, f6_AB)
    world_B_in_B = apply_F6_transform(world_B, f6_AB)
    world_C_in_B = apply_F6_transform(world_C, f6_AB)
    print(f"\n步骤4：变换到相机B坐标系")
    print(f"A在B中: [{world_A_in_B[0]:.1f}, {world_A_in_B[1]:.1f}, {world_A_in_B[2]:.1f}]")
    print(f"B在B中: [{world_B_in_B[0]:.1f}, {world_B_in_B[1]:.1f}, {world_B_in_B[2]:.1f}]")
    print(f"C在B中: [{world_C_in_B[0]:.1f}, {world_C_in_B[1]:.1f}, {world_C_in_B[2]:.1f}]")
    
    # 步骤5：投影到相机B的图像坐标
    A_img_B = (world_A_in_B[0] * id_B / world_A_in_B[2] + CX,
               world_A_in_B[1] * id_B / world_A_in_B[2] + CY)
    B_img_B = (world_B_in_B[0] * id_B / world_B_in_B[2] + CX,
               world_B_in_B[1] * id_B / world_B_in_B[2] + CY)
    C_img_B = (world_C_in_B[0] * id_B / world_C_in_B[2] + CX,
               world_C_in_B[1] * id_B / world_C_in_B[2] + CY)
    print(f"\n步骤5：投影到相机B图像坐标")
    print(f"A: ({A_img_B[0]:.1f}, {A_img_B[1]:.1f})")
    print(f"B: ({B_img_B[0]:.1f}, {B_img_B[1]:.1f})")
    print(f"C: ({C_img_B[0]:.1f}, {C_img_B[1]:.1f})")
    
    # 验证环节：用相机A和相机B的图像坐标分别反投影计算F6
    print(f"\n=== 验证环节 ===")
    
    # 相机A的F6计算
    f6_A_calc = backproject_to_F6(A_img_A, B_img_A, C_img_A, id_A)
    print(f"相机A的F6: [{f6_A_calc[0]:.1f}, {f6_A_calc[1]:.1f}, {f6_A_calc[2]:.1f}, {f6_A_calc[3]:.1f}, {f6_A_calc[4]:.1f}, {f6_A_calc[5]:.1f}]")
    
    # 相机B的F6计算
    f6_B_calc = backproject_to_F6(A_img_B, B_img_B, C_img_B, id_B)
    print(f"相机B的F6: [{f6_B_calc[0]:.1f}, {f6_B_calc[1]:.1f}, {f6_B_calc[2]:.1f}, {f6_B_calc[3]:.1f}, {f6_B_calc[4]:.1f}, {f6_B_calc[5]:.1f}]")
    
    # 关键验证：相机B坐标系中的3D坐标应该与F6_B的平移部分一致
    print(f"\n关键验证：")
    print(f"相机B中3D坐标的A点: [{world_A_in_B[0]:.1f}, {world_A_in_B[1]:.1f}, {world_A_in_B[2]:.1f}]")
    print(f"F6_B的前三个参数:     [{f6_B_calc[0]:.1f}, {f6_B_calc[1]:.1f}, {f6_B_calc[2]:.1f}]")
    
    # 计算误差
    translation_error = np.linalg.norm(world_A_in_B - f6_B_calc[:3])
    print(f"平移参数误差: {translation_error:.10f}")
    
    # 验证投影一致性
    print(f"\n投影一致性验证：")
    
    # 用相机A的F6重新投影，应该得到原始图像坐标
    # F6表示靶标→相机的变换，投影时需要逆变换
    from utils import invF6
    f6_A_inv = invF6(f6_A_calc)
    world_A_reproj = apply_F6_transform(world_A, f6_A_inv)
    A_img_A_verify = (world_A_reproj[0] * id_A / world_A_reproj[2] + CX,
                      world_A_reproj[1] * id_A / world_A_reproj[2] + CY)
    
    img_error_A = np.sqrt((A_img_A[0] - A_img_A_verify[0])**2 + (A_img_A[1] - A_img_A_verify[1])**2)
    print(f"相机A重投影误差: {img_error_A:.10f} 像素")
    
    # 用相机B的F6重新投影
    f6_B_inv = invF6(f6_B_calc)
    world_A_in_B_reproj = apply_F6_transform(world_A_in_B, f6_B_inv)
    A_img_B_verify = (world_A_in_B_reproj[0] * id_B / world_A_in_B_reproj[2] + CX,
                      world_A_in_B_reproj[1] * id_B / world_A_in_B_reproj[2] + CY)
    
    img_error_B = np.sqrt((A_img_B[0] - A_img_B_verify[0])**2 + (A_img_B[1] - A_img_B_verify[1])**2)
    print(f"相机B重投影误差: {img_error_B:.10f} 像素")
    
    # 判断成功
    success = (translation_error < 1e-6) and (img_error_A < 1e-6) and (img_error_B < 1e-6)
    
    print(f"\n=== 测试结果 ===")
    print(f"平移参数误差: {translation_error:.10f} {'✓' if translation_error < 1e-6 else '✗'}")
    print(f"相机A重投影误差: {img_error_A:.10f} {'✓' if img_error_A < 1e-6 else '✗'}")
    print(f"相机B重投影误差: {img_error_B:.10f} {'✓' if img_error_B < 1e-6 else '✗'}")
    
    return success

if __name__ == "__main__":
    success = correct_dual_camera_test()
    print(f"\n结果: {'🎉 完美成功！' if success else '❌ 仍有问题'}")
    
    if success:
        print(f"\n✅ 验证了正确的双相机标定流程：")
        print(f"1. 直接用F6_AB变换3D坐标 ✓")
        print(f"2. 投影-反投影完全一致 ✓")
        print(f"3. 误差在浮点精度范围内 ✓")