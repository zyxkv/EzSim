#!/usr/bin/env python3
"""
生成基础几何体障碍物的OBJ文件
包括sphere，box，cylinder，三棱锥，四棱锥，四棱台，圆台等基础几何体
所有几何体保持单位尺寸，便于后续使用时进行scale拉伸
"""
import math
import numpy as np
import os



def generate_sphere_obj(filename: str, radius: float = 1.0, segments: int = 32, rings: int = 16):
    """生成球体的OBJ文件（包含UV坐标）"""
    vertices = []
    uv_coords = []
    faces = []
    
    # 生成顶点和UV坐标
    for ring in range(rings + 1):
        phi = math.pi * ring / rings  # 从0到pi
        v = ring / rings  # V坐标从0到1
        
        for seg in range(segments):
            theta = 2 * math.pi * seg / segments  # 从0到2pi
            u = seg / segments  # U坐标从0到1
            
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertices.append([x, y, z])
            uv_coords.append([u, v])
    
    # 生成面
    for ring in range(rings):
        for seg in range(segments):
            # 当前环的索引
            curr = ring * segments + seg + 1  # OBJ索引从1开始
            next_seg = ring * segments + ((seg + 1) % segments) + 1
            next_ring = (ring + 1) * segments + seg + 1
            next_ring_next = (ring + 1) * segments + ((seg + 1) % segments) + 1
            
            if ring == 0:  # 顶部三角形
                faces.append([(curr, curr), (next_ring_next, next_ring_next), (next_ring, next_ring)])
            elif ring == rings - 1:  # 底部三角形
                faces.append([(curr, curr), (next_seg, next_seg), (next_ring, next_ring)])
            else:  # 中间四边形分为两个三角形
                faces.append([(curr, curr), (next_seg, next_seg), (next_ring_next, next_ring_next)])
                faces.append([(next_ring_next, next_ring_next), (next_ring, next_ring), (curr, curr)])
    
    # 写入OBJ文件
    with open(filename, 'w') as f:
        f.write("# Sphere Model with UV coordinates\n")
        f.write(f"# Radius: {radius}, Segments: {segments}, Rings: {rings}\n\n")
        
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        # 写入UV坐标
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")
        
        # 写入面（包含UV坐标引用）
        for face in faces:
            face_str = "f"
            for vertex_idx, uv_idx in face:
                face_str += f" {vertex_idx}/{uv_idx}"
            f.write(face_str + "\n")
    
    print(f"Generated sphere model with UV coordinates: {filename}")

def generate_box_obj(filename: str, width: float = 1.0, height: float = 1.0, depth: float = 1.0):
    """生成立方体的OBJ文件（包含UV坐标）"""
    half_w, half_h, half_d = width/2, height/2, depth/2
    
    # 8个顶点
    vertices = [
        [-half_w, -half_h, -half_d],  # 0: 左下前
        [half_w, -half_h, -half_d],   # 1: 右下前
        [half_w, half_h, -half_d],    # 2: 右上前
        [-half_w, half_h, -half_d],   # 3: 左上前
        [-half_w, -half_h, half_d],   # 4: 左下后
        [half_w, -half_h, half_d],    # 5: 右下后
        [half_w, half_h, half_d],     # 6: 右上后
        [-half_w, half_h, half_d],    # 7: 左上后
    ]
    
    # UV坐标 (每个面使用独立的UV坐标)
    uv_coords = [
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # 基本UV四边形
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # 重复使用
    ]
    
    # 面定义（包含顶点索引和UV索引）
    # 格式：[顶点索引/UV索引, 顶点索引/UV索引, 顶点索引/UV索引]
    faces = [
        # 前面 (Z-)
        [(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (1, 1)],
        # 后面 (Z+)
        [(5, 1), (8, 4), (7, 3)], [(7, 3), (6, 2), (5, 1)],
        # 左面 (X-)
        [(4, 1), (8, 4), (5, 1)], [(5, 1), (1, 1), (4, 4)],
        # 右面 (X+)
        [(2, 1), (6, 2), (7, 3)], [(7, 3), (3, 4), (2, 1)],
        # 底面 (Y-)
        [(1, 1), (5, 2), (6, 3)], [(6, 3), (2, 4), (1, 1)],
        # 顶面 (Y+)
        [(4, 1), (3, 2), (7, 3)], [(7, 3), (8, 4), (4, 1)],
    ]
    
    with open(filename, 'w') as f:
        f.write("# Box Model with UV coordinates\n")
        f.write(f"# Dimensions: {width} x {height} x {depth}\n\n")
        
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        # 写入UV坐标
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")
        
        # 写入面（包含UV坐标引用）
        for face in faces:
            face_str = "f"
            for vertex_idx, uv_idx in face:
                face_str += f" {vertex_idx}/{uv_idx}"
            f.write(face_str + "\n")
    
    print(f"Generated box model with UV coordinates: {filename}")

def generate_cylinder_obj(filename: str, radius: float = 1.0, height: float = 2.0, segments: int = 32):
    """生成圆柱体的OBJ文件（包含UV坐标）"""
    vertices = []
    uv_coords = []
    faces = []
    half_height = height / 2
    
    # 底面圆心和顶面圆心
    vertices.append([0, 0, -half_height])  # 0: 底面中心
    vertices.append([0, 0, half_height])   # 1: 顶面中心
    uv_coords.append([0.5, 0.5])  # 底面中心UV
    uv_coords.append([0.5, 0.5])  # 顶面中心UV
    
    # 底面圆周顶点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append([x, y, -half_height])  # 底面圆周
        # 底面UV坐标（径向分布）
        u = 0.5 + 0.5 * math.cos(angle)
        v = 0.5 + 0.5 * math.sin(angle)
        uv_coords.append([u, v])
    
    # 顶面圆周顶点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append([x, y, half_height])   # 顶面圆周
        # 顶面UV坐标（径向分布）
        u = 0.5 + 0.5 * math.cos(angle)
        v = 0.5 + 0.5 * math.sin(angle)
        uv_coords.append([u, v])
    
    # 侧面UV坐标（展开的圆柱面）
    side_uv_start = len(uv_coords)
    for i in range(segments):
        u = i / segments  # 环绕方向
        uv_coords.append([u, 0.0])  # 底边
        uv_coords.append([u, 1.0])  # 顶边
    
    # 生成面
    # 底面三角形
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([(1, 1), (2 + next_i + 1, 2 + next_i + 1), (2 + i + 1, 2 + i + 1)])
    
    # 顶面三角形
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([(2, 2), (2 + segments + i + 1, 2 + segments + i + 1), (2 + segments + next_i + 1, 2 + segments + next_i + 1)])
    
    # 侧面四边形（分为两个三角形，使用展开的UV坐标）
    for i in range(segments):
        next_i = (i + 1) % segments
        bottom_curr = 2 + i + 1
        bottom_next = 2 + next_i + 1
        top_curr = 2 + segments + i + 1
        top_next = 2 + segments + next_i + 1
        
        # 使用侧面专用UV坐标
        uv_bottom_curr = side_uv_start + i * 2 + 1
        uv_bottom_next = side_uv_start + next_i * 2 + 1
        uv_top_curr = side_uv_start + i * 2 + 2
        uv_top_next = side_uv_start + next_i * 2 + 2
        
        faces.append([(bottom_curr, uv_bottom_curr), (bottom_next, uv_bottom_next), (top_next, uv_top_next)])
        faces.append([(top_next, uv_top_next), (top_curr, uv_top_curr), (bottom_curr, uv_bottom_curr)])
    
    with open(filename, 'w') as f:
        f.write("# Cylinder Model with UV coordinates\n")
        f.write(f"# Radius: {radius}, Height: {height}, Segments: {segments}\n\n")
        
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        # 写入UV坐标
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")
        
        # 写入面
        for face in faces:
            face_str = "f"
            for vertex_idx, uv_idx in face:
                face_str += f" {vertex_idx}/{uv_idx}"
            f.write(face_str + "\n")
    
    print(f"Generated cylinder model with UV coordinates: {filename}")

def generate_triangular_pyramid_obj(filename: str, base_size: float = 1.0, height: float = 1.0):
    """生成三棱锥（四面体）的OBJ文件"""
    half_base = base_size / 2
    # 正三角形底面的高
    tri_height = base_size * math.sqrt(3) / 2
    
    vertices = [
        [0, -tri_height / 3, 0],           # 0: 底面中心前
        [-half_base, tri_height * 2 / 3, 0],  # 1: 底面左后
        [half_base, tri_height * 2 / 3, 0],   # 2: 底面右后
        [0, 0, height],                    # 3: 顶点
    ]
    
    faces = [
        [1, 2, 3],  # 底面
        [1, 4, 2],  # 前面
        [2, 4, 3],  # 右面
        [3, 4, 1],  # 左面
    ]
    
    with open(filename, 'w') as f:
        f.write("# Triangular Pyramid Model\n")
        f.write(f"# Base size: {base_size}, Height: {height}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Generated triangular pyramid model: {filename}")

def generate_square_pyramid_obj(filename: str, base_size: float = 1.0, height: float = 1.0):
    """生成四棱锥的OBJ文件"""
    half_base = base_size / 2
    
    vertices = [
        [-half_base, -half_base, 0],  # 0: 底面左前
        [half_base, -half_base, 0],   # 1: 底面右前
        [half_base, half_base, 0],    # 2: 底面右后
        [-half_base, half_base, 0],   # 3: 底面左后
        [0, 0, height],               # 4: 顶点
    ]
    
    faces = [
        [1, 2, 3], [3, 4, 1],  # 底面（两个三角形）
        [1, 5, 2],             # 前面
        [2, 5, 3],             # 右面
        [3, 5, 4],             # 后面
        [4, 5, 1],             # 左面
    ]
    
    with open(filename, 'w') as f:
        f.write("# Square Pyramid Model\n")
        f.write(f"# Base size: {base_size}, Height: {height}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Generated square pyramid model: {filename}")

def generate_frustum_obj(filename: str, bottom_size: float = 1.0, top_size: float = 0.5, height: float = 1.0):
    """生成四棱台的OBJ文件"""
    half_bottom = bottom_size / 2
    half_top = top_size / 2
    half_height = height / 2
    
    vertices = [
        # 底面
        [-half_bottom, -half_bottom, -half_height],  # 0: 底面左前
        [half_bottom, -half_bottom, -half_height],   # 1: 底面右前
        [half_bottom, half_bottom, -half_height],    # 2: 底面右后
        [-half_bottom, half_bottom, -half_height],   # 3: 底面左后
        # 顶面
        [-half_top, -half_top, half_height],         # 4: 顶面左前
        [half_top, -half_top, half_height],          # 5: 顶面右前
        [half_top, half_top, half_height],           # 6: 顶面右后
        [-half_top, half_top, half_height],          # 7: 顶面左后
    ]
    
    faces = [
        # 底面
        [1, 2, 3], [3, 4, 1],
        # 顶面
        [5, 8, 7], [7, 6, 5],
        # 侧面
        [1, 5, 6], [6, 2, 1],  # 前面
        [2, 6, 7], [7, 3, 2],  # 右面
        [3, 7, 8], [8, 4, 3],  # 后面
        [4, 8, 5], [5, 1, 4],  # 左面
    ]
    
    with open(filename, 'w') as f:
        f.write("# Square Frustum Model\n")
        f.write(f"# Bottom: {bottom_size}, Top: {top_size}, Height: {height}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Generated square frustum model: {filename}")

def generate_cone_frustum_obj(filename: str, bottom_radius: float = 1.0, top_radius: float = 0.5, 
                             height: float = 2.0, segments: int = 32):
    """生成圆台的OBJ文件"""
    vertices = []
    faces = []
    half_height = height / 2
    
    # 底面中心和顶面中心
    vertices.append([0, 0, -half_height])  # 0: 底面中心
    vertices.append([0, 0, half_height])   # 1: 顶面中心
    
    # 底面圆周顶点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = bottom_radius * math.cos(angle)
        y = bottom_radius * math.sin(angle)
        vertices.append([x, y, -half_height])  # 2 + i
    
    # 顶面圆周顶点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = top_radius * math.cos(angle)
        y = top_radius * math.sin(angle)
        vertices.append([x, y, half_height])   # 2 + segments + i
    
    # 底面三角形
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([1, 2 + next_i + 1, 2 + i + 1])
    
    # 顶面三角形
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([2, 2 + segments + i + 1, 2 + segments + next_i + 1])
    
    # 侧面梯形（分为两个三角形）
    for i in range(segments):
        next_i = (i + 1) % segments
        bottom_curr = 2 + i + 1
        bottom_next = 2 + next_i + 1
        top_curr = 2 + segments + i + 1
        top_next = 2 + segments + next_i + 1
        
        faces.append([bottom_curr, bottom_next, top_next])
        faces.append([top_next, top_curr, bottom_curr])
    
    with open(filename, 'w') as f:
        f.write("# Cone Frustum Model\n")
        f.write(f"# Bottom radius: {bottom_radius}, Top radius: {top_radius}\n")
        f.write(f"# Height: {height}, Segments: {segments}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Generated cone frustum model: {filename}")

def main():
    pi_seg = 128
    """主函数：生成所有基础几何体障碍物"""
    # 创建输出目录
    output_dir = "simple_obstacles"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating basic geometry obstacle models...")
    
    # 生成球体 (半径1.0，使用适中的精度)
    generate_sphere_obj(os.path.join(output_dir, "sphere.obj"), 
                       radius=1.0, segments=pi_seg, rings=pi_seg//2)
    
    # 生成立方体 (1x1x1)
    generate_box_obj(os.path.join(output_dir, "box.obj"))
    
    # 生成圆柱体 (半径1.0，高度2.0)
    generate_cylinder_obj(os.path.join(output_dir, "cylinder.obj"), 
                         radius=1.0, height=2.0, segments=pi_seg)
    
    # 生成三棱锥 (底边长1.0，高度1.0)
    generate_triangular_pyramid_obj(os.path.join(output_dir, "triangular_pyramid.obj"))
    
    # 生成四棱锥 (底边长1.0，高度1.0)
    generate_square_pyramid_obj(os.path.join(output_dir, "square_pyramid.obj"))
    
    # 生成四棱台 (底边长1.0，顶边长0.5，高度1.0)
    generate_frustum_obj(os.path.join(output_dir, "square_frustum.obj"))
    
    # 生成圆台 (底半径1.0，顶半径0.5，高度2.0)
    generate_cone_frustum_obj(os.path.join(output_dir, "cone_frustum.obj"), 
                             bottom_radius=1.0, top_radius=0.5, height=2.0, segments=pi_seg)
    
    print(f"\n所有障碍物模型已保存到 '{output_dir}' 目录:")
    print(f"- sphere.obj (球体)")
    print(f"- box.obj (立方体)")
    print(f"- cylinder.obj (圆柱体)")
    print(f"- triangular_pyramid.obj (三棱锥)")
    print(f"- square_pyramid.obj (四棱锥)")
    print(f"- square_frustum.obj (四棱台)")
    print(f"- cone_frustum.obj (圆台)")
    print("\n所有模型均为单位尺寸，可在使用时通过scale参数进行拉伸。")
    print(f"圆形相关几何体使用{pi_seg}段采样，平衡了精确度和渲染效率。")

if __name__ == "__main__":
    main()
