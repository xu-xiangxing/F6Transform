#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from dstrtCrrct import DistortionCorrection

def verify_320_points():
    """验证320个点的提取和排序"""
    print("=== 验证320个质心点 ===")
    
    # 创建校正器并提取点
    corrector = DistortionCorrection("../images/dstrt.jpg")
    
    # 加载原图和二值化图像
    enhanced = corrector.load_and_preprocess()
    binary = corrector.load_ideal_binary()
    
    # 提取320个质心（包含假畸变校正排序）
    centers = corrector.extract_grid_points(binary)
    
    if len(centers) != 320:
        print(f"❌ 点数不对: {len(centers)}")
        return
    
    print(f"✅ 成功提取320个点")
    
    # 加载原图
    original_pil = Image.open("../images/dstrt.jpg").convert('RGB')
    draw = ImageDraw.Draw(original_pil)
    
    # 在原图上标记320个点
    for i, (x, y) in enumerate(centers):
        # 用小圆圈标记每个点
        radius = 2
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill='red', outline='red')
        
        # 每50个点用不同颜色
        if i % 50 == 0:
            draw.ellipse([x-3, y-3, x+3, y+3], 
                        fill='yellow', outline='yellow')
    
    # 保存标记后的图像
    original_pil.save('../images/points_marked.jpg')
    print("✅ 320个点已标记到原图，保存为 points_marked.jpg")
    
    # 建立x,y,u,v数组
    create_mapping_arrays(corrector)

def create_mapping_arrays(corrector):
    """建立x,y,u,v映照数组"""
    print("\n=== 建立x,y,u,v映照数组 ===")
    
    if not hasattr(corrector, 'sorted_grid_16x20'):
        print("❌ 没有找到16x20排序结果")
        return
    
    # 提取畸变坐标 (x,y) 和假校正坐标 (u,v)
    x_distorted = []
    y_distorted = []
    u_corrected = []
    v_corrected = []
    
    # 生成理想的16x20网格作为u,v
    ideal_spacing_x = 30  # 理想网格X间距
    ideal_spacing_y = 25  # 理想网格Y间距
    start_x = 50
    start_y = 50
    
    for row in range(16):
        for col in range(20):
            # 从排序好的网格中获取畸变坐标
            distorted_point = corrector.sorted_grid_16x20[row][col]
            x_distorted.append(distorted_point[0])
            y_distorted.append(distorted_point[1])
            
            # 计算理想网格坐标
            ideal_u = start_x + col * ideal_spacing_x
            ideal_v = start_y + row * ideal_spacing_y
            u_corrected.append(ideal_u)
            v_corrected.append(ideal_v)
    
    # 转换为numpy数组
    x_array = np.array(x_distorted)
    y_array = np.array(y_distorted)
    u_array = np.array(u_corrected)
    v_array = np.array(v_corrected)
    
    print(f"✅ 建立映照数组完成:")
    print(f"   x范围: {x_array.min():.1f} - {x_array.max():.1f}")
    print(f"   y范围: {y_array.min():.1f} - {y_array.max():.1f}")
    print(f"   u范围: {u_array.min():.1f} - {u_array.max():.1f}")
    print(f"   v范围: {v_array.min():.1f} - {v_array.max():.1f}")
    
    # 保存映照数组
    np.savez('../data/mapping_arrays.npz', 
             x=x_array, y=y_array, u=u_array, v=v_array)
    print("✅ 映照数组已保存为 mapping_arrays.npz")
    
    # 可视化理想网格
    visualize_ideal_grid(u_array, v_array)
    
    return x_array, y_array, u_array, v_array

def visualize_ideal_grid(u_array, v_array):
    """可视化理想网格"""
    print("\n=== 可视化理想网格 ===")
    
    # 创建理想网格图像
    max_u = int(u_array.max()) + 50
    max_v = int(v_array.max()) + 50
    
    ideal_image = Image.new('RGB', (max_u, max_v), 'white')
    draw = ImageDraw.Draw(ideal_image)
    
    # 绘制320个理想网格点
    for u, v in zip(u_array, v_array):
        radius = 3
        draw.ellipse([u-radius, v-radius, u+radius, v+radius], 
                    fill='blue', outline='blue')
    
    # 保存理想网格图像
    ideal_image.save('../images/ideal_grid.jpg')
    print("✅ 理想网格已保存为 ideal_grid.jpg")

if __name__ == "__main__":
    verify_320_points()