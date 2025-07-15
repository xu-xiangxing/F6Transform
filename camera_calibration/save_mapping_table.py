#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from dstrtCrrct import DistortionCorrection

def save_mapping_table_to_file():
    """生成并保存畸变到理想的变换表"""
    
    print("=== 生成xy2uv变换表 ===")
    
    # 创建畸变校正器
    corrector = DistortionCorrection("dstrt.jpg")
    
    # 运行到生成映照表的步骤
    try:
        # 1-7步：生成映照表
        enhanced = corrector.load_and_preprocess()
        binary = corrector.load_ideal_binary()
        grid_points = corrector.extract_grid_points(binary)
        sorted_grid = corrector.sort_grid_points()
        cells = corrector.create_cells()
        ideal_grid = corrector.generate_ideal_grid()
        mapping_table = corrector.create_mapping_table()
        
        print(f"成功生成映照表，包含 {len(mapping_table)} 个控制点")
        
        # 保存为JSON格式
        save_mapping_as_json(mapping_table, "xy2uv_mapping.json")
        
        # 保存为numpy格式（更高效）
        save_mapping_as_numpy(mapping_table, "xy2uv_mapping.npz")
        
        # 分析并保存统计信息
        analyze_mapping_table(mapping_table, corrector)
        
        return True
        
    except Exception as e:
        print(f"生成映照表失败: {e}")
        return False

def save_mapping_as_json(mapping_table, filename):
    """保存映照表为JSON格式"""
    
    # 将映照表转换为JSON可序列化的格式
    json_mapping = {}
    
    for (dist_x, dist_y), (ideal_x, ideal_y) in mapping_table.items():
        # 使用字符串作为键
        key = f"{dist_x},{dist_y}"
        json_mapping[key] = [float(ideal_x), float(ideal_y)]
    
    # 保存到文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "description": "畸变坐标到理想坐标的映射表",
            "format": "key: 'x,y' -> value: [ideal_x, ideal_y]",
            "total_points": len(json_mapping),
            "mapping": json_mapping
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON格式映照表已保存: {filename}")

def save_mapping_as_numpy(mapping_table, filename):
    """保存映照表为numpy格式"""
    
    # 转换为numpy数组
    distorted_points = []
    ideal_points = []
    
    for (dist_x, dist_y), (ideal_x, ideal_y) in mapping_table.items():
        distorted_points.append([dist_x, dist_y])
        ideal_points.append([ideal_x, ideal_y])
    
    distorted_array = np.array(distorted_points)
    ideal_array = np.array(ideal_points)
    
    # 保存为.npz文件
    np.savez(filename,
             distorted_points=distorted_array,
             ideal_points=ideal_array,
             total_points=len(mapping_table))
    
    print(f"✓ NumPy格式映照表已保存: {filename}")

def analyze_mapping_table(mapping_table, corrector):
    """分析映照表并保存统计信息"""
    
    print("\n=== 映照表分析 ===")
    
    # 提取坐标
    distorted_coords = np.array(list(mapping_table.keys()))
    ideal_coords = np.array(list(mapping_table.values()))
    
    # 计算统计信息
    stats = {
        "total_control_points": len(mapping_table),
        "distorted_range": {
            "x_min": float(distorted_coords[:, 0].min()),
            "x_max": float(distorted_coords[:, 0].max()),
            "y_min": float(distorted_coords[:, 1].min()),
            "y_max": float(distorted_coords[:, 1].max())
        },
        "ideal_range": {
            "x_min": float(ideal_coords[:, 0].min()),
            "x_max": float(ideal_coords[:, 0].max()),
            "y_min": float(ideal_coords[:, 1].min()),
            "y_max": float(ideal_coords[:, 1].max())
        }
    }
    
    # 计算位移统计
    displacement = ideal_coords - distorted_coords
    displacement_magnitude = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
    
    stats["displacement_stats"] = {
        "mean_displacement": float(displacement_magnitude.mean()),
        "max_displacement": float(displacement_magnitude.max()),
        "min_displacement": float(displacement_magnitude.min()),
        "std_displacement": float(displacement_magnitude.std())
    }
    
    # 分析网格信息
    if hasattr(corrector, 'sorted_grid_16x20') and corrector.sorted_grid_16x20:
        grid = corrector.sorted_grid_16x20
        stats["grid_info"] = {
            "rows": len(grid),
            "cols": len(grid[0]) if grid else 0,
            "total_grid_points": sum(len(row) for row in grid)
        }
    
    # 保存统计信息
    with open("mapping_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 映照表分析已保存: mapping_analysis.json")
    
    # 打印摘要
    print(f"控制点数量: {stats['total_control_points']}")
    print(f"平均位移: {stats['displacement_stats']['mean_displacement']:.2f} 像素")
    print(f"最大位移: {stats['displacement_stats']['max_displacement']:.2f} 像素")

def load_and_test_mapping(filename="xy2uv_mapping.npz"):
    """加载并测试映照表"""
    
    print(f"\n=== 测试加载映照表: {filename} ===")
    
    try:
        # 加载numpy格式
        data = np.load(filename)
        distorted_points = data['distorted_points']
        ideal_points = data['ideal_points']
        total_points = data['total_points']
        
        print(f"成功加载 {total_points} 个映射点")
        
        # 显示前5个映射关系
        print("前5个映射关系:")
        for i in range(min(5, len(distorted_points))):
            dist_x, dist_y = distorted_points[i]
            ideal_x, ideal_y = ideal_points[i]
            print(f"  ({dist_x:.0f}, {dist_y:.0f}) -> ({ideal_x:.1f}, {ideal_y:.1f})")
        
        return distorted_points, ideal_points
        
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None

def create_interpolation_function(distorted_points, ideal_points):
    """创建插值函数用于任意点的坐标变换"""
    
    from scipy.interpolate import griddata
    
    def xy2uv_transform(x, y):
        """将畸变坐标(x,y)变换为理想坐标(u,v)"""
        
        # 使用最近邻插值（快速）
        query_point = np.array([[x, y]])
        
        # 分别插值x和y坐标
        u = griddata(distorted_points, ideal_points[:, 0], query_point, method='nearest')[0]
        v = griddata(distorted_points, ideal_points[:, 1], query_point, method='nearest')[0]
        
        return u, v
    
    return xy2uv_transform

def main():
    """主函数"""
    
    print("生成xy2uv变换表程序")
    print("=" * 50)
    
    # 1. 生成并保存映照表
    success = save_mapping_table_to_file()
    
    if success:
        print("\n" + "=" * 50)
        
        # 2. 测试加载
        distorted_pts, ideal_pts = load_and_test_mapping()
        
        if distorted_pts is not None:
            # 3. 创建变换函数
            transform_func = create_interpolation_function(distorted_pts, ideal_pts)
            
            # 4. 测试变换
            print("\n=== 测试坐标变换 ===")
            test_points = [(100, 100), (500, 300), (800, 600)]
            
            for x, y in test_points:
                u, v = transform_func(x, y)
                print(f"畸变点 ({x}, {y}) -> 理想点 ({u:.1f}, {v:.1f})")
        
        print("\n生成的文件:")
        print("- xy2uv_mapping.json: JSON格式映照表")
        print("- xy2uv_mapping.npz: NumPy格式映照表")
        print("- mapping_analysis.json: 统计分析结果")
    
    else:
        print("映照表生成失败")

if __name__ == "__main__":
    main()