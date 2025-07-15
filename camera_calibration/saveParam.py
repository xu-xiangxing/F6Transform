#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
from dstrtCrrct import DistortionCorrection

def save_system_parameters():
    """保存系统参数：320个节点数据 + 285个grid数据"""
    print("=== 保存系统参数 ===")
    
    # 创建校正器并运行完整流程
    corrector = DistortionCorrection("dstrt.jpg")
    
    # 运行完整处理流程
    enhanced = corrector.load_and_preprocess()
    binary = corrector.load_ideal_binary()
    centers = corrector.extract_grid_points(binary)
    sorted_grid = corrector.sort_grid_points()
    cells = corrector.create_cells()
    ideal_grid = corrector.generate_ideal_grid()
    
    # 1. 保存320个节点数据
    save_node_data(corrector)
    
    # 2. 保存285个grid数据  
    save_grid_data(corrector)
    
    # 3. 保存完整的系统配置
    save_system_config(corrector)
    
    print("✅ 系统参数保存完成")

def save_node_data(corrector):
    """保存320个节点数据"""
    print("\n--- 保存320个节点数据 ---")
    
    if not hasattr(corrector, 'sorted_grid_16x20'):
        print("❌ 没有找到16x20排序数据")
        return
    
    # 收集所有节点数据
    nodes_data = []
    node_id = 0
    
    for row in range(16):
        for col in range(20):
            point = corrector.sorted_grid_16x20[row][col]
            
            node_data = {
                'id': node_id,
                'row': row,
                'col': col,
                'x_distorted': float(point[0]),  # 畸变图像坐标
                'y_distorted': float(point[1]),
                'x_ideal': 50.0 + col * 30.0,    # 理想网格坐标
                'y_ideal': 50.0 + row * 25.0
            }
            
            nodes_data.append(node_data)
            node_id += 1
    
    # 保存为JSON格式
    with open('nodes_320.json', 'w', encoding='utf-8') as f:
        json.dump({
            'description': '320个网格节点数据 (16行x20列)',
            'total_nodes': len(nodes_data),
            'grid_size': {'rows': 16, 'cols': 20},
            'nodes': nodes_data
        }, f, indent=2, ensure_ascii=False)
    
    # 保存为numpy格式（便于快速加载）
    x_distorted = [node['x_distorted'] for node in nodes_data]
    y_distorted = [node['y_distorted'] for node in nodes_data]
    x_ideal = [node['x_ideal'] for node in nodes_data]
    y_ideal = [node['y_ideal'] for node in nodes_data]
    
    np.savez('nodes_320.npz',
             x_distorted=np.array(x_distorted),
             y_distorted=np.array(y_distorted),
             x_ideal=np.array(x_ideal),
             y_ideal=np.array(y_ideal),
             rows=np.array([node['row'] for node in nodes_data]),
             cols=np.array([node['col'] for node in nodes_data]))
    
    print(f"✅ 320个节点数据已保存:")
    print(f"   nodes_320.json  (人类可读)")
    print(f"   nodes_320.npz   (快速加载)")

def save_grid_data(corrector):
    """保存285个grid数据 (xy畸变四边形 + uv理想正方形)"""
    print("\n--- 保存285个grid数据 ---")
    
    if not hasattr(corrector, 'sorted_grid_16x20'):
        print("❌ 没有找到网格数据")
        return
    
    # 收集所有grid cell数据
    grids_data = []
    grid_id = 0
    
    for row in range(15):  # 15行grid
        for col in range(19):  # 19列grid
            # 获取畸变四边形的4个角点 (x,y坐标)
            xy_top_left = corrector.sorted_grid_16x20[row][col]
            xy_top_right = corrector.sorted_grid_16x20[row][col + 1]
            xy_bottom_right = corrector.sorted_grid_16x20[row + 1][col + 1]
            xy_bottom_left = corrector.sorted_grid_16x20[row + 1][col]
            
            # 计算理想正方形的4个角点 (u,v坐标)
            uv_top_left = (50.0 + col * 30.0, 50.0 + row * 25.0)
            uv_top_right = (50.0 + (col + 1) * 30.0, 50.0 + row * 25.0)
            uv_bottom_right = (50.0 + (col + 1) * 30.0, 50.0 + (row + 1) * 25.0)
            uv_bottom_left = (50.0 + col * 30.0, 50.0 + (row + 1) * 25.0)
            
            grid_data = {
                'id': grid_id,
                'row': row,
                'col': col,
                'xy_distorted_quad': {
                    'x1': float(xy_top_left[0]), 'y1': float(xy_top_left[1]),
                    'x2': float(xy_top_right[0]), 'y2': float(xy_top_right[1]),
                    'x3': float(xy_bottom_right[0]), 'y3': float(xy_bottom_right[1]),
                    'x4': float(xy_bottom_left[0]), 'y4': float(xy_bottom_left[1])
                },
                'uv_ideal_square': {
                    'u1': uv_top_left[0], 'v1': uv_top_left[1],
                    'u2': uv_top_right[0], 'v2': uv_top_right[1],
                    'u3': uv_bottom_right[0], 'v3': uv_bottom_right[1],
                    'u4': uv_bottom_left[0], 'v4': uv_bottom_left[1]
                }
            }
            
            grids_data.append(grid_data)
            grid_id += 1
    
    # 保存为JSON格式
    with open('grids_285.json', 'w', encoding='utf-8') as f:
        json.dump({
            'description': '285个grid数据：xy畸变四边形 + uv理想正方形 (15行x19列)',
            'total_grids': len(grids_data),
            'grid_size': {'rows': 15, 'cols': 19},
            'coordinate_system': {
                'xy': '畸变图像坐标系 (真实四边形)',
                'uv': '理想校正坐标系 (标准正方形)'
            },
            'grids': grids_data
        }, f, indent=2, ensure_ascii=False)
    
    # 同时保存为numpy格式便于算法使用
    x_coords = []
    y_coords = []
    u_coords = []
    v_coords = []
    
    for grid in grids_data:
        xy = grid['xy_distorted_quad']
        uv = grid['uv_ideal_square']
        
        # 每个grid的4个角点
        x_coords.extend([xy['x1'], xy['x2'], xy['x3'], xy['x4']])
        y_coords.extend([xy['y1'], xy['y2'], xy['y3'], xy['y4']])
        u_coords.extend([uv['u1'], uv['u2'], uv['u3'], uv['u4']])
        v_coords.extend([uv['v1'], uv['v2'], uv['v3'], uv['v4']])
    
    np.savez('grids_285.npz',
             x_distorted=np.array(x_coords),
             y_distorted=np.array(y_coords),
             u_ideal=np.array(u_coords),
             v_ideal=np.array(v_coords),
             grid_ids=np.repeat(range(285), 4))  # 每个grid重复4次
    
    print(f"✅ 285个grid数据已保存:")
    print(f"   grids_285.json  (xy畸变四边形 + uv理想正方形)")
    print(f"   grids_285.npz   (numpy格式，便于算法使用)")

def save_system_config(corrector):
    """保存系统配置参数"""
    print("\n--- 保存系统配置 ---")
    
    config = {
        'system_info': {
            'description': '相机畸变校正系统参数',
            'algorithm': '假畸变校正 + 射线重定向',
            'grid_structure': '16x20节点, 15x19四边形',
            'total_nodes': 320,
            'total_grids': 285
        },
        'image_info': {
            'source_image': 'dstrt.jpg',
            'binary_image': 'ideal_binary.jpg',
            'image_size': [640, 480]
        },
        'grid_parameters': {
            'node_rows': 16,
            'node_cols': 20,
            'grid_rows': 15,
            'grid_cols': 19,
            'ideal_spacing_x': 30.0,
            'ideal_spacing_y': 25.0,
            'ideal_start_x': 50.0,
            'ideal_start_y': 50.0
        },
        'algorithm_parameters': {
            'fake_correction_center': [338.00, 241.90],
            'compensation_factor': 0.001,
            'area_filter_range': [50, 250]
        }
    }
    
    with open('system_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 系统配置已保存:")
    print(f"   system_config.json")

if __name__ == "__main__":
    save_system_parameters()