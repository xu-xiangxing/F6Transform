#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils import calc_image_depth, backproject, compute_relative_F6, apply_F6_transform, combine_F6
from backPrjct import backproject_to_F6

def test_step1_basic_setup():
    """第一步：定义两个相机的id参数和A相机屏幕上的三角形，执行反投影"""
    print("=== 第一步：基础设置和反投影测试 ===")
    
    # 定义两个相机的id参数
    print("\n1. 定义相机参数:")
    width = 1920
    fov_A = 60.0
    fov_B = 58.0
    id_A = calc_image_depth(fov_A, width)
    id_B = calc_image_depth(fov_B, width)
    
    print(f"相机A - FOV: {fov_A}°, 像深id_A: {id_A:.1f} 像素")
    print(f"相机B - FOV: {fov_B}°, 像深id_B: {id_B:.1f} 像素")
    
    # 定义A相机屏幕上任意三角形坐标
    print("\n2. 定义A相机观察到的三角形图像坐标:")
    A_img_A = (960.0, 700.0)   # 图像中心偏下
    B_img_A = (800.0, 540.0)   # 左上
    C_img_A = (1120.0, 540.0)  # 右上
    
    print(f"A点图像坐标: {A_img_A}")
    print(f"B点图像坐标: {B_img_A}")
    print(f"C点图像坐标: {C_img_A}")
    
    # 执行反投影并打印数据
    print("\n3. 执行反投影，获得3D世界坐标:")
    triangle_3d = backproject(A_img_A, B_img_A, C_img_A, id_A)
    
    # 提取并打印3D坐标
    world_A = np.array(triangle_3d[0])
    world_B = np.array(triangle_3d[1]) 
    world_C = np.array(triangle_3d[2])
    
    print(f"A点3D坐标: [{world_A[0]:.3f}, {world_A[1]:.3f}, {world_A[2]:.3f}] mm")
    print(f"B点3D坐标: [{world_B[0]:.3f}, {world_B[1]:.3f}, {world_B[2]:.3f}] mm")
    print(f"C点3D坐标: [{world_C[0]:.3f}, {world_C[1]:.3f}, {world_C[2]:.3f}] mm")
    
    # 计算三角形边长验证
    print("\n4. 验证三角形几何特性:")
    edge_AB = np.linalg.norm(world_A - world_B)
    edge_BC = np.linalg.norm(world_B - world_C)
    edge_CA = np.linalg.norm(world_C - world_A)
    
    print(f"边长AB: {edge_AB:.3f} mm")
    print(f"边长BC: {edge_BC:.3f} mm")
    print(f"边长CA: {edge_CA:.3f} mm")
    
    # 计算F6参数
    print("\n5. 从A相机图像坐标反投影计算F6_A:")
    f6_A = backproject_to_F6(A_img_A, B_img_A, C_img_A, id_A)
    print(f"F6_A: [{f6_A[0]:.3f}, {f6_A[1]:.3f}, {f6_A[2]:.3f}, {f6_A[3]:.3f}, {f6_A[4]:.3f}, {f6_A[5]:.3f}]")
    
    # 在B相机定义不同的三角形（略有差异，避免重合）
    print("\n6. 定义B相机观察到的三角形图像坐标（与A相机略有差异）:")
    A_img_B = (950.0, 720.0)   # 与A相机略有差异
    B_img_B = (790.0, 550.0)   # 略有差异
    C_img_B = (1130.0, 530.0)  # 略有差异
    
    print(f"A点图像坐标: {A_img_B}")
    print(f"B点图像坐标: {B_img_B}")
    print(f"C点图像坐标: {C_img_B}")
    
    # 从B相机反投影得到3D坐标
    print("\n7. B相机反投影得到3D世界坐标:")
    triangle_3d_B = backproject(A_img_B, B_img_B, C_img_B, id_B)
    
    world_A_B = np.array(triangle_3d_B[0])
    world_B_B = np.array(triangle_3d_B[1])
    world_C_B = np.array(triangle_3d_B[2])
    
    print(f"A点3D坐标: [{world_A_B[0]:.3f}, {world_A_B[1]:.3f}, {world_A_B[2]:.3f}] mm")
    print(f"B点3D坐标: [{world_B_B[0]:.3f}, {world_B_B[1]:.3f}, {world_B_B[2]:.3f}] mm") 
    print(f"C点3D坐标: [{world_C_B[0]:.3f}, {world_C_B[1]:.3f}, {world_C_B[2]:.3f}] mm")
    
    # 验证B相机反投影的三角形几何特性
    print("\n8. 验证B相机反投影的三角形几何特性:")
    edge_AB_B = np.linalg.norm(world_A_B - world_B_B)
    edge_BC_B = np.linalg.norm(world_B_B - world_C_B)
    edge_CA_B = np.linalg.norm(world_C_B - world_A_B)
    
    print(f"边长AB: {edge_AB_B:.3f} mm")
    print(f"边长BC: {edge_BC_B:.3f} mm")
    print(f"边长CA: {edge_CA_B:.3f} mm")
    
    # 计算B相机的F6参数
    print("\n9. 从B相机图像坐标反投影计算F6_B:")
    f6_B = backproject_to_F6(A_img_B, B_img_B, C_img_B, id_B)
    print(f"F6_B: [{f6_B[0]:.3f}, {f6_B[1]:.3f}, {f6_B[2]:.3f}, {f6_B[3]:.3f}, {f6_B[4]:.3f}, {f6_B[5]:.3f}]")
    
    # 计算相对变换 F6_AB = F6_A ⊕ !F6_B
    print("\n10. 计算相对变换 F6_AB = F6_A ⊕ !F6_B:")
    f6_AB = compute_relative_F6(f6_A, f6_B)
    print(f"F6_AB: [{f6_AB[0]:.3f}, {f6_AB[1]:.3f}, {f6_AB[2]:.3f}, {f6_AB[3]:.3f}, {f6_AB[4]:.3f}, {f6_AB[5]:.3f}]")
    print("(这表示从相机A坐标系到相机B坐标系的变换)")
    print("解释：同一个靶标被两个相机观察，F6_AB使得'B看到的就是A看到的'")
    
    # 总结F6三者关系
    print("\n=== F6三者关系总结 ===")
    print("已知任意两个，可求第三个：")
    print("1. F6_AB = F6_A ⊕ !F6_B  (已知F6_A和F6_B，求F6_AB)")
    print("2. F6_B = !F6_AB ⊕ F6_A  (已知F6_A和F6_AB，求F6_B)")
    print("3. F6_A = F6_AB ⊕ F6_B   (已知F6_B和F6_AB，求F6_A)")
    print("这就是双相机标定的核心原理！")
    
    # 测试质点变换：验证B看到的变成A看到的
    print("\n=== 质点变换测试 ===")
    print("验证：通过F6_AB变换，B看到的质点变成A看到的")
    
    # 在B相机坐标系中定义一个测试点
    test_point_B = np.array([10.0, 20.0, 100.0])  # B相机坐标系中的点
    print(f"测试点在B相机坐标系中: [{test_point_B[0]:.1f}, {test_point_B[1]:.1f}, {test_point_B[2]:.1f}] mm")
    
    # 方法1：直接使用apply_F6_transform
    print("\n方法1：使用apply_F6_transform（内部将质点转换为F6）")
    test_point_A_method1 = apply_F6_transform(test_point_B, f6_AB)
    print(f"变换后在A相机坐标系中: [{test_point_A_method1[0]:.1f}, {test_point_A_method1[1]:.1f}, {test_point_A_method1[2]:.1f}] mm")
    
    # 方法2：显式将质点转换为F6，然后用F6群运算
    print("\n方法2：显式质点→F6，然后F6群运算")
    f6_point_B = np.concatenate([test_point_B, np.array([0.0, 0.0, 0.0])])  # 质点作为F6
    print(f"质点的F6表示: [{f6_point_B[0]:.1f}, {f6_point_B[1]:.1f}, {f6_point_B[2]:.1f}, {f6_point_B[3]:.1f}, {f6_point_B[4]:.1f}, {f6_point_B[5]:.1f}]")
    
    # F6群运算：F6_AB ⊕ F6_point_B
    f6_point_A = combine_F6(f6_AB, f6_point_B)
    test_point_A_method2 = f6_point_A[:3]  # 提取位置分量
    print(f"F6运算结果: [{f6_point_A[0]:.1f}, {f6_point_A[1]:.1f}, {f6_point_A[2]:.1f}, {f6_point_A[3]:.1f}, {f6_point_A[4]:.1f}, {f6_point_A[5]:.1f}]")
    print(f"提取位置分量: [{test_point_A_method2[0]:.1f}, {test_point_A_method2[1]:.1f}, {test_point_A_method2[2]:.1f}] mm")
    
    # 验证两种方法结果一致
    diff = np.linalg.norm(test_point_A_method1 - test_point_A_method2)
    print(f"\n两种方法的差异: {diff:.6f} mm")
    if diff < 1e-6:
        print("✓ 两种方法结果一致，验证了质点变换的F6群内运算本质！")
    
    return {
        'id_A': id_A,
        'id_B': id_B,
        'img_coords_A': (A_img_A, B_img_A, C_img_A),
        'img_coords_B': (A_img_B, B_img_B, C_img_B),
        'world_coords_A': (world_A, world_B, world_C),
        'world_coords_B': (world_A_B, world_B_B, world_C_B),
        'f6_A': f6_A,
        'f6_B': f6_B,
        'f6_AB': f6_AB
    }

if __name__ == "__main__":
    print("双相机测试 - 第一步：基础反投影")
    print("=" * 50)
    
    result = test_step1_basic_setup()
    
    print("\n" + "=" * 50)
    print("第一步完成！数据已准备好供下一步使用。")