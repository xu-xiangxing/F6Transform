#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from scipy.ndimage import convolve, uniform_filter, binary_closing, binary_opening, sobel, gaussian_filter, maximum_filter

class DistortionCorrection:
    """畸变图像的射线重定向校正"""
    
    def __init__(self, image_path, binary_path=None):
        self.image_path = image_path
        self.binary_path = binary_path or "../images/ideal_binary.jpg"
        self.original_image = None
        self.binary_image = None
        self.grid_points = []
        self.sorted_grid = None
        self.cells = []
        self.ideal_grid = []
        self.mapping_table = {}
        
    def load_and_preprocess(self):
        """1. 加载图像并预处理"""
        print("=== 1. 图像加载和预处理 ===")
        
        # 加载图像
        try:
            pil_image = Image.open(self.image_path)
            gray_image = pil_image.convert('L')
            self.original_image = np.array(gray_image)
        except Exception as e:
            raise ValueError(f"无法加载图像: {self.image_path}, 错误: {e}")
        
        height, width = self.original_image.shape
        print(f"图像尺寸: {width} x {height}")
        
        # 简单的图像增强
        enhanced = self.simple_histogram_equalization(self.original_image)
        
        # 简单高斯滤波
        blurred = self.simple_gaussian_blur(enhanced)
        
        return blurred
    
    def load_ideal_binary(self):
        """2. 加载理想二值化图像"""
        print("=== 2. 加载理想二值化图像 ===")
        
        try:
            # 加载理想的二值化图像
            pil_image = Image.open(self.binary_path)
            gray_image = pil_image.convert('L')
            binary_image = np.array(gray_image)
            
            # 确保是真正的二值图像
            binary_image = (binary_image > 128).astype(np.uint8) * 255
            
            self.binary_image = binary_image
            print(f"成功加载理想二值化图像: {self.binary_path}")
            print(f"图像尺寸: {binary_image.shape[1]} x {binary_image.shape[0]}")
            
            return binary_image
            
        except Exception as e:
            print(f"加载理想二值化图像失败: {e}")
            return None
    
    def extract_grid_points(self, binary_image):
        """3. 质点化 - 从连通区域提取圆点中心"""
        print("=== 3. 质点化 - 从连通区域提取圆点中心 ===")
        
        # 使用连通区域分析提取真正的圆点中心
        corners = self.extract_circle_centers(binary_image)
        
        if corners is not None and len(corners) > 0:
            # 转换为列表格式
            self.grid_points = [(float(x), float(y)) for x, y in corners]
        else:
            self.grid_points = []
            
        print(f"检测到网格点数量: {len(self.grid_points)}")
        
        return self.grid_points
    
    def sort_grid_points(self):
        """4. 使用假畸变校正的排序结果"""
        print("=== 4. 使用假畸变校正排序结果 ===")
        
        if hasattr(self, 'sorted_grid_16x20') and self.sorted_grid_16x20:
            self.sorted_grid = self.sorted_grid_16x20
            print(f"✅ 使用16x20排序结果，共 {len(self.sorted_grid)} 行")
            print(f"✅ 每行 {len(self.sorted_grid[0])} 个点")
            
            # 验证
            total_points = sum(len(row) for row in self.sorted_grid)
            print(f"✅ 总点数: {total_points}")
            
            return self.sorted_grid
        else:
            print("❌ 没有找到假畸变校正排序结果")
            return None
    
    def create_cells(self):
        """5. 创建四边形cell"""
        print("=== 5. 创建四边形单元 ===")
        
        if not self.sorted_grid or len(self.sorted_grid) < 2:
            return []
        
        cells = []
        
        for row_idx in range(len(self.sorted_grid) - 1):
            current_row = self.sorted_grid[row_idx]
            next_row = self.sorted_grid[row_idx + 1]
            
            # 确保两行长度足够
            min_length = min(len(current_row), len(next_row))
            
            for col_idx in range(min_length - 1):
                try:
                    # 四个角点：左上、右上、右下、左下
                    top_left = current_row[col_idx]
                    top_right = current_row[col_idx + 1]
                    bottom_right = next_row[col_idx + 1]
                    bottom_left = next_row[col_idx]
                    
                    cell = [top_left, top_right, bottom_right, bottom_left]
                    cells.append(cell)
                    
                except IndexError:
                    continue
        
        self.cells = cells
        print(f"创建了 {len(cells)} 个四边形单元")
        
        return cells
    
    def generate_ideal_grid(self):
        """6. 生成对应的理想网格"""
        print("=== 6. 生成理想网格 ===")
        
        if not self.cells:
            return []
        
        # 计算网格的行列数
        if not self.sorted_grid:
            return []
        
        rows = len(self.sorted_grid)
        cols = len(self.sorted_grid[0]) if self.sorted_grid else 0
        
        print(f"网格规模: {rows} 行 x {cols} 列")
        
        # 设置理想网格的参数
        cell_size = 50  # 理想网格单元大小（像素）
        start_x = 100   # 起始X坐标
        start_y = 100   # 起始Y坐标
        
        ideal_grid = []
        
        for row in range(rows):
            ideal_row = []
            for col in range(cols):
                ideal_x = start_x + col * cell_size
                ideal_y = start_y + row * cell_size
                ideal_row.append((ideal_x, ideal_y))
            ideal_grid.append(ideal_row)
        
        self.ideal_grid = ideal_grid
        print(f"理想网格生成完成")
        
        return ideal_grid
    
    def create_mapping_table(self):
        """7. 创建映照表"""
        print("=== 7. 创建映照表 ===")
        
        if not self.cells or not self.ideal_grid:
            return {}
        
        mapping_table = {}
        cell_mappings = []
        
        # 为每个cell创建映照关系
        for cell_idx, distorted_cell in enumerate(self.cells):
            # 计算对应的理想cell
            row_idx = cell_idx // (len(self.sorted_grid[0]) - 1) if self.sorted_grid else 0
            col_idx = cell_idx % (len(self.sorted_grid[0]) - 1) if self.sorted_grid else 0
            
            if (row_idx < len(self.ideal_grid) - 1 and 
                col_idx < len(self.ideal_grid[0]) - 1):
                
                ideal_cell = [
                    self.ideal_grid[row_idx][col_idx],       # 左上
                    self.ideal_grid[row_idx][col_idx + 1],   # 右上
                    self.ideal_grid[row_idx + 1][col_idx + 1], # 右下
                    self.ideal_grid[row_idx + 1][col_idx]    # 左下
                ]
                
                cell_mappings.append((distorted_cell, ideal_cell))
        
        # 存储cell映照关系
        self.cell_mappings = cell_mappings
        print(f"创建了 {len(cell_mappings)} 个cell映照关系")
        
        # 为控制点创建直接映照
        for distorted_cell, ideal_cell in cell_mappings:
            for dist_pt, ideal_pt in zip(distorted_cell, ideal_cell):
                key = (int(round(dist_pt[0])), int(round(dist_pt[1])))
                mapping_table[key] = ideal_pt
        
        self.mapping_table = mapping_table
        print(f"映照表包含 {len(mapping_table)} 个控制点")
        
        return mapping_table
    
    def bilinear_interpolation(self, x, y):
        """8. 双线性插值"""
        # 找到包含该点的cell
        for distorted_cell, ideal_cell in getattr(self, 'cell_mappings', []):
            if self.point_in_quad(x, y, distorted_cell):
                return self.bilinear_interp_in_quad(x, y, distorted_cell, ideal_cell)
        
        # 如果没有找到包含的cell，使用最近邻插值
        return self.nearest_neighbor_interpolation(x, y)
    
    def point_in_quad(self, x, y, quad):
        """判断点是否在四边形内"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        point = (x, y)
        
        # 使用三角形法判断
        d1 = sign(point, quad[0], quad[1])
        d2 = sign(point, quad[1], quad[2])
        d3 = sign(point, quad[2], quad[3])
        d4 = sign(point, quad[3], quad[0])
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
        
        return not (has_neg and has_pos)
    
    def bilinear_interp_in_quad(self, x, y, src_quad, dst_quad):
        """在四边形内进行双线性插值"""
        # 简化处理：使用重心坐标
        # 这是一个简化的实现，实际中可能需要更精确的算法
        
        # 计算到四个顶点的距离的倒数作为权重
        weights = []
        for pt in src_quad:
            dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
            weights.append(1.0 / (dist + 1e-6))  # 避免除零
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 加权平均得到目标坐标
        result_x = sum(w * dst_pt[0] for w, dst_pt in zip(weights, dst_quad))
        result_y = sum(w * dst_pt[1] for w, dst_pt in zip(weights, dst_quad))
        
        return (result_x, result_y)
    
    def nearest_neighbor_interpolation(self, x, y):
        """最近邻插值"""
        if not self.mapping_table:
            return (x, y)
        
        # 找到最近的映照点
        min_dist = float('inf')
        nearest_ideal = (x, y)
        
        for (map_x, map_y), ideal_pt in self.mapping_table.items():
            dist = (x - map_x)**2 + (y - map_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_ideal = ideal_pt
        
        return nearest_ideal
    
    def simple_histogram_equalization(self, image):
        """简单的直方图均衡化"""
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(image.shape).astype(np.uint8)
    
    def simple_gaussian_blur(self, image, kernel_size=5):
        """简单的高斯模糊"""
        # 创建简单的高斯核
        sigma = kernel_size / 6.0
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2)/(2*sigma**2)),
            (kernel_size, kernel_size)
        )
        kernel = kernel / kernel.sum()
        
        # 应用卷积
        from scipy.ndimage import convolve
        return convolve(image, kernel).astype(np.uint8)
    
    def simple_adaptive_threshold(self, image, block_size=11, C=2):
        """简单的自适应阈值"""
        from scipy.ndimage import uniform_filter
        # 计算局部均值
        local_mean = uniform_filter(image.astype(np.float32), size=block_size)
        # 阈值化
        binary = (image > local_mean - C).astype(np.uint8) * 255
        return binary
    
    def simple_morphology(self, binary_image):
        """简单的形态学操作"""
        from scipy.ndimage import binary_closing, binary_opening
        # 闭运算
        closed = binary_closing(binary_image > 128, structure=np.ones((3,3)))
        # 开运算
        opened = binary_opening(closed, structure=np.ones((3,3)))
        return (opened * 255).astype(np.uint8)
    
    def simple_corner_detection(self, binary_image):
        """简化的网格点检测"""
        print("使用简化的网格点检测方法...")
        
        # 直接模拟网格点（基于图像特征的合理假设）
        height, width = binary_image.shape
        
        # 假设网格大小
        rows = 15
        cols = 20
        margin = 50
        
        corners = []
        for row in range(rows):
            for col in range(cols):
                # 基本网格位置
                x = margin + col * (width - 2*margin) // (cols - 1)
                y = margin + row * (height - 2*margin) // (rows - 1)
                
                # 添加一些随机性来模拟真实检测
                x += np.random.randint(-5, 6)
                y += np.random.randint(-5, 6)
                
                # 确保在图像范围内
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                corners.append((x, y))
        
        print(f"生成了 {len(corners)} 个模拟网格点")
        return corners
    
    def block_binarization(self, image, blocks=(8, 8)):
        """8x8分块二值化处理"""
        print(f"开始{blocks[0]}x{blocks[1]}分块二值化...")
        
        height, width = image.shape
        block_h = height // blocks[0]
        block_w = width // blocks[1]
        
        # 创建输出图像
        binary_image = np.zeros_like(image)
        
        # 计算边缘无效区域（约10%边缘）
        edge_margin_h = int(height * 0.1)
        edge_margin_w = int(width * 0.1)
        
        print(f"图像尺寸: {width}x{height}")
        print(f"块大小: {block_w}x{block_h}")
        print(f"边缘无效区域: {edge_margin_w}x{edge_margin_h}")
        
        for row in range(blocks[0]):
            for col in range(blocks[1]):
                # 计算块的边界
                start_y = row * block_h
                end_y = min((row + 1) * block_h, height)
                start_x = col * block_w  
                end_x = min((col + 1) * block_w, width)
                
                # 提取当前块
                block = image[start_y:end_y, start_x:end_x]
                
                # 判断是否为边缘块
                is_edge_block = (start_y < edge_margin_h or 
                               end_y > height - edge_margin_h or
                               start_x < edge_margin_w or 
                               end_x > width - edge_margin_w)
                
                if is_edge_block:
                    # 边缘块：使用更保守的阈值处理
                    threshold = self.calculate_robust_threshold(block)
                    print(f"边缘块 [{row},{col}]: 阈值 {threshold:.1f}")
                else:
                    # 中心块：使用标准阈值处理
                    threshold = self.calculate_standard_threshold(block)
                    print(f"中心块 [{row},{col}]: 阈值 {threshold:.1f}")
                
                # 应用阈值
                block_binary = (block > threshold).astype(np.uint8) * 255
                
                # 将结果写回原图
                binary_image[start_y:end_y, start_x:end_x] = block_binary
        
        print("分块二值化完成")
        return binary_image
    
    def calculate_robust_threshold(self, block):
        """计算边缘块的保守阈值"""
        # 使用中位数作为基准，避免极值影响
        median = np.median(block)
        std = np.std(block)
        
        # 边缘区域使用更保守的阈值
        threshold = median + 0.5 * std
        return threshold
    
    def calculate_standard_threshold(self, block):
        """计算中心块的标准阈值"""
        # 使用均值和标准差
        mean = np.mean(block)
        std = np.std(block)
        
        # 中心区域可以使用更激进的阈值
        threshold = mean + 0.3 * std
        return threshold
    
    def connected_component_filtering(self, binary_image):
        """连通区域分析和体积过滤"""
        print("开始连通区域分析和体积过滤...")
        
        from scipy.ndimage import label, center_of_mass
        
        # 连通区域标记
        labeled_image, num_features = label(binary_image)
        print(f"检测到 {num_features} 个连通区域")
        
        # 分析每个连通区域的特征
        valid_regions = []
        height, width = binary_image.shape
        
        # 估计圆点的理想大小范围
        expected_dot_area = (width * height) / (20 * 15)  # 假设20x15网格
        min_area = expected_dot_area * 0.01  # 最小1%，更宽松
        max_area = expected_dot_area * 20.0  # 最大2000%，更宽松
        
        print(f"预期圆点面积: {expected_dot_area:.0f}")
        print(f"面积过滤范围: {min_area:.0f} - {max_area:.0f}")
        
        for region_id in range(1, num_features + 1):
            # 获取当前区域的像素
            region_mask = (labeled_image == region_id)
            region_area = np.sum(region_mask)
            
            # 面积过滤
            if min_area <= region_area <= max_area:
                # 计算区域的紧凑度（圆度）
                compactness = self.calculate_compactness(region_mask)
                
                # 圆度过滤 - 更宽松
                if compactness > 0.1:  # 更宽松的形状要求
                    valid_regions.append(region_id)
        
        print(f"面积+圆度过滤后剩余: {len(valid_regions)} 个区域")
        
        # 创建过滤后的二值图像
        filtered_image = np.zeros_like(binary_image)
        for region_id in valid_regions:
            filtered_image[labeled_image == region_id] = 255
        
        return filtered_image
    
    def calculate_compactness(self, region_mask):
        """计算区域的紧凑度（圆度）"""
        # 计算周长和面积
        area = np.sum(region_mask)
        
        # 简单的边缘检测来计算周长
        from scipy.ndimage import sobel
        edges = sobel(region_mask.astype(np.uint8))
        perimeter = np.sum(edges > 0)
        
        if perimeter == 0:
            return 0
        
        # 紧凑度 = 4π * 面积 / 周长²
        # 完美的圆的紧凑度为1
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        return compactness
    
    def extract_circle_centers(self, binary_image):
        """从理想二值化图像提取圆点中心"""
        print("开始从连通区域提取圆点中心...")
        
        from scipy.ndimage import label, center_of_mass
        
        # 连通区域标记
        labeled_image, num_features = label(binary_image)
        print(f"检测到 {num_features} 个连通区域")
        
        # 根据实际观察的圆点面积范围调整
        # 从输出看，正常圆点面积在77-188像素范围
        min_area = 50   # 最小50像素
        max_area = 250  # 最大250像素
        expected_area = (min_area + max_area) / 2  # 估算平均面积
        
        print(f"期望圆点面积: {expected_area:.0f} 像素")
        print(f"面积过滤范围: {min_area:.0f} - {max_area:.0f} 像素")
        
        # 提取符合条件的连通区域中心
        circle_centers = []
        filtered_count = 0
        
        for region_id in range(1, num_features + 1):
            # 计算当前区域的面积
            region_mask = (labeled_image == region_id)
            region_area = np.sum(region_mask)
            
            # 面积过滤
            if min_area <= region_area <= max_area:
                # 获取当前区域的质心
                center = center_of_mass(binary_image, labeled_image, region_id)
                
                if center is not None:
                    y, x = center  # center_of_mass返回(row, col)格式
                    circle_centers.append((x, y))
            else:
                filtered_count += 1
                print(f"过滤区域 {region_id}: 面积 {region_area} 超出范围")
        
        print(f"过滤掉 {filtered_count} 个不符合面积要求的区域")
        print(f"成功提取 {len(circle_centers)} 个圆点中心")
        print(f"理论值应该是 16x20=320 个点")
        
        if len(circle_centers) == 320:
            print("✅ 点数完全匹配！")
        else:
            print(f"⚠️  点数不匹配！检测到{len(circle_centers)}个，期望320个")
        
        # 验证质心准确性
        self.verify_center_accuracy(circle_centers, labeled_image, binary_image)
        
        # 应用"假畸变校正"进行排序
        sorted_centers = self.fake_distortion_correction_sort(circle_centers)
        
        return sorted_centers
    
    def verify_center_accuracy(self, centers, labeled_image, binary_image):
        """验证质心是否准确对应圆点中心"""
        print("\n=== 质心准确性验证 ===")
        
        # 分析质心分布
        if len(centers) == 0:
            print("❌ 没有检测到质心")
            return
        
        centers_array = np.array(centers)
        x_coords = centers_array[:, 0]
        y_coords = centers_array[:, 1]
        
        print(f"X坐标范围: {x_coords.min():.1f} - {x_coords.max():.1f}")
        print(f"Y坐标范围: {y_coords.min():.1f} - {y_coords.max():.1f}")
        print(f"X坐标中位数: {np.median(x_coords):.1f}")
        print(f"Y坐标中位数: {np.median(y_coords):.1f}")
        
        # 检查几个采样点的质心精度
        print("\n采样质心验证（前10个点）:")
        for i in range(min(10, len(centers))):
            x, y = centers[i]
            print(f"  点{i+1}: ({x:.2f}, {y:.2f})")
            
            # 检查该点周围的像素分布
            self.analyze_center_region(x, y, binary_image, radius=5)
    
    def analyze_center_region(self, center_x, center_y, binary_image, radius=5):
        """分析质心周围区域的像素分布"""
        height, width = binary_image.shape
        
        # 获取周围区域
        x = int(round(center_x))
        y = int(round(center_y))
        
        # 边界检查
        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)
        
        region = binary_image[y1:y2, x1:x2]
        
        # 分析区域
        white_pixels = np.sum(region > 128)
        total_pixels = region.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        # 简单的对称性检查
        center_pixel = binary_image[y, x] if 0 <= x < width and 0 <= y < height else 0
        
        status = "✅" if white_ratio > 0.5 and center_pixel > 128 else "⚠️"
        print(f"    {status} 中心像素值: {center_pixel}, 周围白像素比: {white_ratio:.2f}")
        
        return white_ratio > 0.5
    
    def fake_distortion_correction_sort(self, centers):
        """假畸变校正排序算法"""
        print("\n=== 假畸变校正排序 ===")
        
        if len(centers) != 320:
            print(f"❌ 点数不是320个，无法进行16x20排序")
            return centers
        
        # 1. 计算质点群中心
        centers_array = np.array(centers)
        center_x = np.mean(centers_array[:, 0])
        center_y = np.mean(centers_array[:, 1])
        print(f"质点群中心: ({center_x:.2f}, {center_y:.2f})")
        
        # 2. 计算每个点到中心的射线向量和距离
        flattened_points = []
        for x, y in centers:
            # 计算到中心的向量
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # 径向补偿：距离越远，延长越多
            # 这里使用简单的线性补偿，实际可能需要调整
            compensation_factor = 1.0 + distance * 0.001  # 补偿系数
            
            # 计算拉平后的坐标
            flat_x = center_x + dx * compensation_factor
            flat_y = center_y + dy * compensation_factor
            
            flattened_points.append((flat_x, flat_y, x, y))  # (拉平x, 拉平y, 原x, 原y)
        
        print("径向补偿拉平完成")
        
        # 3. 在拉平后的坐标上按Y坐标排序
        flattened_points.sort(key=lambda p: p[1])  # 按拉平后的Y坐标排序
        
        # 4. 每20个点分为一行，共16行
        sorted_grid = []
        for row in range(16):
            start_idx = row * 20
            end_idx = start_idx + 20
            row_points = flattened_points[start_idx:end_idx]
            
            # 每行内部按拉平后的X坐标排序
            row_points.sort(key=lambda p: p[0])
            
            # 提取原始坐标
            row_original = [(p[2], p[3]) for p in row_points]
            sorted_grid.append(row_original)
        
        print(f"✅ 成功排序为16行x20列网格")
        
        # 验证排序结果
        total_points = sum(len(row) for row in sorted_grid)
        print(f"排序后总点数: {total_points}")
        
        # 将16x20的格式转换回普通列表，保持原有接口兼容
        sorted_centers = []
        for row in sorted_grid:
            sorted_centers.extend(row)
        
        # 保存排序好的网格结构供后续使用
        self.sorted_grid_16x20 = sorted_grid
        
        return sorted_centers
    
    def apply_correction(self, output_size=None):
        """应用畸变校正"""
        print("=== 9. 应用畸变校正 ===")
        
        if self.original_image is None:
            return None
        
        height, width = self.original_image.shape
        if output_size is None:
            output_size = (width, height)
        
        # 创建校正后的图像
        corrected_image = np.zeros(output_size[::-1], dtype=np.uint8)
        
        print("正在进行像素重映射...")
        
        # 简化版本：只处理采样点
        sample_step = 10  # 大幅降采样加速
        for y in range(0, output_size[1], sample_step):
            for x in range(0, output_size[0], sample_step):
                # 找到对应的原图坐标
                corrected_coord = self.bilinear_interpolation(x, y)
                
                src_x, src_y = corrected_coord
                src_x, src_y = int(round(src_x)), int(round(src_y))
                
                # 检查边界
                if 0 <= src_x < width and 0 <= src_y < height:
                    # 填充采样区域
                    for dy in range(sample_step):
                        for dx in range(sample_step):
                            if y+dy < output_size[1] and x+dx < output_size[0]:
                                corrected_image[y+dy, x+dx] = self.original_image[src_y, src_x]
        
        print("畸变校正完成")
        return corrected_image
    
    def run_complete_correction(self):
        """运行完整的畸变校正流程"""
        print("*" * 60)
        print("真实畸变图像的射线重定向校正")
        print("*" * 60)
        
        try:
            # 1. 加载和预处理
            enhanced = self.load_and_preprocess()
            
            # 2. 加载理想二值化图像
            binary = self.load_ideal_binary()
            
            # 3. 提取网格点
            grid_points = self.extract_grid_points(binary)
            
            # 4. 排序
            sorted_grid = self.sort_grid_points()
            
            # 5. 创建cells
            cells = self.create_cells()
            
            # 6. 生成理想网格
            ideal_grid = self.generate_ideal_grid()
            
            # 7. 创建映照表
            mapping_table = self.create_mapping_table()
            
            # 8. 应用校正
            corrected_image = self.apply_correction()
            
            # 保存结果
            if corrected_image is not None:
                Image.fromarray(corrected_image).save('../images/corrected_image.jpg')
                print("校正后图像已保存为: corrected_image.jpg")
            
            # 保存中间结果
            if self.binary_image is not None:
                Image.fromarray(self.binary_image).save('../images/binary_image.jpg')
                print("二值化图像已保存为: binary_image.jpg")
            
            print("\n=== 射线重定向校正完成 ===")
            return True
            
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            return False

def test_real_distortion_correction():
    """测试真实畸变校正"""
    
    # 创建校正器
    corrector = DistortionCorrection("../images/dstrt.jpg")
    
    # 运行完整流程
    success = corrector.run_complete_correction()
    
    if success:
        print("\n✓ 真实畸变图像的射线重定向校正成功完成！")
        print("这演示了:")
        print("  1. 真实畸变图像的处理能力")
        print("  2. 网格检测和质点化技术")
        print("  3. 四边形cell的映照建立")
        print("  4. 双线性插值的校正效果")
        print("  5. 射线重定向的实际应用")
    else:
        print("✗ 校正过程遇到问题")

if __name__ == "__main__":
    test_real_distortion_correction()