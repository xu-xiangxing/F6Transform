#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw

def draw_ideal_grid(rows=16, cols=20, cell_size=50, margin=100, image_size=(1280, 1024)):
    """绘制理想的网格图像"""
    
    # 创建白色背景图像
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    
    # 计算起始位置
    start_x = margin
    start_y = margin
    
    # 绘制网格点（圆点）
    dot_radius = 5
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * cell_size
            y = start_y + row * cell_size
            
            # 绘制黑色圆点
            draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], 
                        fill='black')
    
    # 绘制网格线（可选）
    draw_lines = True
    if draw_lines:
        # 横线
        for row in range(rows):
            y = start_y + row * cell_size
            x1 = start_x
            x2 = start_x + (cols-1) * cell_size
            draw.line([(x1, y), (x2, y)], fill='gray', width=1)
        
        # 竖线
        for col in range(cols):
            x = start_x + col * cell_size
            y1 = start_y
            y2 = start_y + (rows-1) * cell_size
            draw.line([(x, y1), (x, y2)], fill='gray', width=1)
    
    # 保存图像
    image.save('../images/ideal_grid.jpg')
    print(f"理想网格图已保存: ideal_grid.jpg")
    print(f"网格规模: {rows}x{cols}")
    print(f"单元格大小: {cell_size}像素")
    print(f"图像尺寸: {image_size}")
    
    return image

def draw_ideal_grid_binary(rows=16, cols=20, cell_size=50, margin=100, image_size=(1280, 1024)):
    """绘制理想的二值化网格图（只有圆点）"""
    
    # 创建白色背景图像
    image = Image.new('L', image_size, 255)  # L模式为灰度图
    draw = ImageDraw.Draw(image)
    
    # 计算起始位置
    start_x = margin
    start_y = margin
    
    # 绘制网格点（黑色圆点）
    dot_radius = 8  # 稍大一些的圆点
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * cell_size
            y = start_y + row * cell_size
            
            # 绘制黑色圆点
            draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], 
                        fill=0)  # 0 = 黑色
    
    # 保存图像
    image.save('../images/ideal_binary_grid.jpg')
    print(f"理想二值网格图已保存: ideal_binary_grid.jpg")
    
    return image

if __name__ == "__main__":
    # 生成两种理想网格图
    print("生成理想网格图...")
    
    # 1. 带网格线的理想图
    draw_ideal_grid()
    
    # 2. 只有圆点的二值图
    draw_ideal_grid_binary()
    
    print("\n完成！生成了两个文件：")
    print("- ideal_grid.jpg: 带网格线的理想图")
    print("- ideal_binary_grid.jpg: 只有圆点的二值图")