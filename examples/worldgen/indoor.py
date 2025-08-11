import torch 
import math 
import numpy as np
import argparse
import yaml
import random
from typing import Dict, List, Tuple, Optional

import ezsim
# from ezsim.sensors import SensorDataRecorder, VideoFileWriter
from ezsim.utils.image_exporter import FrameImageExporter

def build_floor(scene: ezsim.Scene, floor_x: float, floor_y: float):
    """构建地板"""
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(floor_x, floor_y, 0.1),
            pos=(0, 0, 0.05),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.6, 0.6, 0.6, 1.0)
        )
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_concrete_floor_02.png"
        #     )
        # )
    )
    return floor_x, floor_y

def build_roof(scene: ezsim.Scene, floor_x: float, floor_y: float, roof_h: float):
    """构建屋顶"""
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(floor_x, floor_y, 0.1),
            pos=(0, 0, roof_h-0.05),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_plastered_wall_04.png"
        #     )
        # )
        surface=ezsim.surfaces.Rough(
            color=(0.7, 0.7, 0.7, 1.0)
        ),
        
    )
    return roof_h

def build_walls(scene: ezsim.Scene, floor_x: float, floor_y: float, roof_h: float):
    """构建四面墙"""
    wall_thickness = 0.2
    
    # 前墙
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(floor_x, wall_thickness, roof_h),
            pos=(0, floor_y/2 - wall_thickness/2, roof_h/2),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.8, 0.6, 0.4, 1.0)
        )
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_brick_wall_001.png"
        #     )
        # )
    )
    
    # 后墙
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(floor_x, wall_thickness, roof_h),
            pos=(0, -floor_y/2 + wall_thickness/2, roof_h/2),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.8, 0.6, 0.4, 1.0)
        )
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_brick_wall_001.png"
        #     )
        # )
    )
    
    # 左墙
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(wall_thickness, floor_y, roof_h),
            pos=(-floor_x/2 + wall_thickness/2, 0, roof_h/2),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.8, 0.6, 0.4, 1.0)
        )
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_brick_wall_003.png"
        #     )
        # )
    )
    
    # 右墙
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(wall_thickness, floor_y, roof_h),
            pos=(floor_x/2 - wall_thickness/2, 0, roof_h/2),
            fixed=True  # 设置为固定，不参与物理仿真
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.8, 0.6, 0.4, 1.0)
        )
        # surface=ezsim.surfaces.Rough(
        #     diffuse_texture=ezsim.textures.ImageTexture(
        #         image_path="textures/indoor_brick_wall_003.png"
        #     )
        # )
    )

def build_light(scene: ezsim.Scene, floor_x: float, floor_y: float, roof_h: float, grid_x: int, grid_y: int):
    """构建室内顶棚照明灯矩阵和墙面补充照明
    
    Args:
        scene: EzSim场景对象
        floor_x: 房间长度
        floor_y: 房间宽度  
        roof_h: 房间高度
        grid_x: X方向灯的数量
        grid_y: Y方向灯的数量
    """
    # 灯的高度：距离顶棚0.5m
    light_height = roof_h - 0.5
    
    # 设置边界距离限制
    x_margin = 5.0  # X方向距离前后边界至少5m
    y_margin = 2.0  # Y方向距离左右边界至少2m
    
    # 计算实际可用的照明区域
    available_x = floor_x - 2 * x_margin
    available_y = floor_y - 2 * y_margin
    
    # 确保有足够的空间放置灯
    if available_x <= 0 or available_y <= 0:
        print(f"Warning: Room too small for lighting with margins. Room: {floor_x}x{floor_y}, Required: {2*x_margin}x{2*y_margin}")
        return
    
    # 计算灯之间的间距和起始位置
    if grid_x > 1:
        x_spacing = available_x / (grid_x - 1)
        x_start = -available_x / 2
    else:
        x_spacing = 0
        x_start = 0
    
    if grid_y > 1:
        y_spacing = available_y / (grid_y - 1)
        y_start = -available_y / 2
    else:
        y_spacing = 0
        y_start = 0
    
    # 天花板灯的基本参数
    light_intensity = 0.1  # 稍微降低主要光源强度
    light_cutoff = 45.0     # 光照范围角度
    
    # 创建天花板灯矩阵
    for i in range(grid_x):
        for j in range(grid_y):
            # 计算灯的位置
            if grid_x > 1:
                x_pos = x_start + i * x_spacing
            else:
                x_pos = 0
                
            if grid_y > 1:
                y_pos = y_start + j * y_spacing
            else:
                y_pos = 0
            
            # 添加点光源
            scene.add_light(
                pos=[x_pos, y_pos, light_height],
                dir=[0.0, 0.0, -1.0],  # 向下照射
                directional=0,         # 点光源
                castshadow=0,          # 产生阴影
                cutoff=light_cutoff,   # 光照范围
                intensity=light_intensity,
            )
            
            # 可选：添加视觉化的灯具实体（小球体表示灯泡）
            scene.add_entity(
                morph=ezsim.morphs.Sphere(
                    radius=0.15,  # 小球体表示灯泡
                    pos=[x_pos, y_pos, light_height + 0.1],
                    fixed=True  # 固定不动
                ),
                surface=ezsim.surfaces.Emission(
                    color=(1.0, 1.0, 0.8, 0.0)  # 淡黄色发光材质
                )
            )
    
    # 添加墙面补充照明以消除光影图案
    wall_light_intensity = 0.1  # 较低的补充光强度
    wall_light_height = roof_h * 0.6  # 墙面灯高度约为房间高度的60%
    wall_offset = 0.5  # 距离墙面的偏移距离
    
    # 前墙补充照明（沿Y正方向）
    wall_light_count = max(3, grid_x // 2)  # 墙面灯数量
    for i in range(wall_light_count):
        x_pos = -floor_x/2 + (i + 1) * floor_x / (wall_light_count + 1)
        scene.add_light(
            pos=[x_pos, floor_y/2 - wall_offset, wall_light_height],
            dir=[0.0, -1.0, 0.0],  # 向房间内照射
            directional=0,
            castshadow=0,  # 不产生阴影，避免额外图案
            cutoff=90.0,   # 较大的照射角度
            intensity=wall_light_intensity,
        )
    
    # 后墙补充照明（沿Y负方向）
    for i in range(wall_light_count):
        x_pos = -floor_x/2 + (i + 1) * floor_x / (wall_light_count + 1)
        scene.add_light(
            pos=[x_pos, -floor_y/2 + wall_offset, wall_light_height],
            dir=[0.0, 1.0, 0.0],  # 向房间内照射
            directional=0,
            castshadow=0,  # 不产生阴影
            cutoff=90.0,
            intensity=wall_light_intensity,
        )
    
    # 左墙补充照明（沿X负方向）  
    wall_light_count_y = max(3, grid_y // 2)
    for i in range(wall_light_count_y):
        y_pos = -floor_y/2 + (i + 1) * floor_y / (wall_light_count_y + 1)
        scene.add_light(
            pos=[-floor_x/2 + wall_offset, y_pos, wall_light_height],
            dir=[1.0, 0.0, 0.0],  # 向房间内照射
            directional=0,
            castshadow=0,  # 不产生阴影
            cutoff=90.0,
            intensity=wall_light_intensity,
        )
    
    # 右墙补充照明（沿X正方向）
    for i in range(wall_light_count_y):
        y_pos = -floor_y/2 + (i + 1) * floor_y / (wall_light_count_y + 1)
        scene.add_light(
            pos=[floor_x/2 - wall_offset, y_pos, wall_light_height],
            dir=[-1.0, 0.0, 0.0],  # 向房间内照射
            directional=0,
            castshadow=0,  # 不产生阴影
            cutoff=90.0,
            intensity=wall_light_intensity,
        )
    
    # 添加一些环境光以进一步平滑光照
    ambient_light_count = 4
    ambient_intensity = 0.05
    for i in range(ambient_light_count):
        angle = i * 2 * np.pi / ambient_light_count
        radius = min(floor_x, floor_y) * 0.3
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)
        
        scene.add_light(
            pos=[x_pos, y_pos, roof_h * 0.8],
            dir=[0.0, 0.0, -1.0],
            directional=0,
            castshadow=0,  # 环境光不产生阴影
            cutoff=90.0,   # 大范围照射
            intensity=ambient_intensity,
        )
    
    total_wall_lights = 2 * wall_light_count + 2 * wall_light_count_y
    total_ceiling_lights = grid_x * grid_y
    
    print(f"Added {total_ceiling_lights} ceiling lights at height {light_height}m")
    print(f"Added {total_wall_lights} wall lights at height {wall_light_height}m for uniform illumination")
    print(f"Added {ambient_light_count} ambient lights to smooth lighting patterns") 

def build_gate(scene: ezsim.Scene, gate_config: dict, roof_h: float) -> List[Dict]:
    """构建门（使用OBJ文件加载）并返回门的占用空间信息"""
    occupied_spaces = []
    
    for gate_name, gate_info in gate_config.items():
        gate_type = gate_info['type']
        pos = gate_info['pos']
        size = gate_info['size']
        
        # 根据方向设置旋转角度
        if gate_info['direction'] == 'x':
            euler = (0.0, 90.0, 0.0)
        elif gate_info['direction'] == 'y':
            euler = (0.0, 90.0, 90.0)
        else:
            raise ValueError(f"Invalid gate direction: {gate_info['direction']}")
        
        if gate_type == 'square':
            # 方形门：配置中size = [宽度, 高度]
            target_width = size[0] if len(size) > 0 else 1.3
            target_height = size[1] if len(size) > 1 else 1.0
            target_depth = size[2] if len(size) > 2 else 0.05

            # 我们生成的方形门OBJ原始尺寸：外框1.3x1.3x0.05
            original_width = 1.3
            original_height = 1.0  # 实际上我们的OBJ没有严格的高度概念，这里是厚度方向
            original_depth = 0.05
            
            # 计算缩放因子
            scale_x = target_width / original_width
            scale_y = target_height / original_height  # Y轴也按宽度比例缩放保持比例
            scale_z = target_depth / original_depth  # Z轴按高度缩放
            
            # 加载方形门OBJ模型
            try:
                scene.add_entity(
                    morph=ezsim.morphs.Mesh(
                        file="meshes/drone_racing/square_gate.obj",  # 使用我们生成的OBJ文件
                        pos=pos,
                        euler=euler,
                        scale=(scale_x, scale_y, scale_z),  # 应用缩放
                        fixed=True
                    ),
                    surface=ezsim.surfaces.Gold(
                        color=(1.0, 0.8, 0.2, 1.0)
                    )
                )
                
                # 记录门的占用空间信息（使用实际配置的尺寸）
                gate_clearance = gate_info.get('clearance', 2.0)
                occupied_spaces.append({
                    'pos': pos,
                    'size': (target_width + gate_clearance, target_width + gate_clearance, target_height),
                    'type': 'gate',
                    'name': gate_name
                })
                
                print(f"Built square gate '{gate_name}' at {pos} with size {target_width}x{target_height}")
                
            except Exception as e:
                print(f"Warning: Failed to load square gate model: {e}")
                continue
                
        elif gate_type == 'circle':
            # 圆形门：配置中size = [半径]
            target_radius = size[0] if len(size) > 0 else 0.7
            target_depth = size[1] if len(size) > 1 else 0.1

            # 我们生成的圆形门OBJ原始尺寸：外半径0.7，内半径0.5，厚度0.1
            original_radius = 0.7
            original_depth = 0.1
            
            # 计算缩放因子
            scale_factor = target_radius / original_radius
            scale_depth = target_depth / original_depth

            # 加载圆形门OBJ模型
            try:
                scene.add_entity(
                    morph=ezsim.morphs.Mesh(
                        file="meshes/drone_racing/circle_gate.obj",  # 使用我们生成的OBJ文件
                        pos=pos,
                        euler=euler,
                        scale=(scale_factor, scale_factor, scale_depth),  # 统一缩放
                        fixed=True
                    ),
                    surface=ezsim.surfaces.Rough(
                        color=(0.8, 0.2, 0.2, 1.0)
                    )
                )
                
                # 记录门的占用空间信息（使用实际配置的尺寸）
                gate_clearance = gate_info.get('clearance', 2.0)
                occupied_spaces.append({
                    'pos': pos,
                    'size': (target_radius*2 + gate_clearance, target_radius*2 + gate_clearance, target_radius*2),
                    'type': 'gate',
                    'name': gate_name
                })
                
                print(f"Built circle gate '{gate_name}' at {pos} with radius {target_radius}")
                
            except Exception as e:
                print(f"Warning: Failed to load circle gate model: {e}")
                continue
    
    return occupied_spaces

def build_obstacles(scene: ezsim.Scene, obstacles_config: List[Dict]):
    """构建各种类型的障碍物"""
    # 为不同类型的障碍物定义颜色
    obstacle_colors = {
        'sphere': [(0.8, 0.4, 0.2, 1.0), (0.2, 0.8, 0.4, 1.0), (0.4, 0.2, 0.8, 1.0)],
        'box': [(0.6, 0.6, 0.2, 1.0), (0.2, 0.6, 0.6, 1.0), (0.6, 0.2, 0.6, 1.0)],
        'cylinder': [(0.8, 0.8, 0.2, 1.0), (0.2, 0.8, 0.8, 1.0), (0.8, 0.2, 0.8, 1.0)]
    }
    
    for i, obstacle in enumerate(obstacles_config):
        obstacle_type = obstacle['type']
        pos = obstacle['pos']
        size = obstacle['size']
        
        # 为每种类型选择颜色
        color_list = obstacle_colors[obstacle_type]
        color = color_list[i % len(color_list)]
        
        if obstacle_type == 'sphere':
            radius = size[0] / 2
            scene.add_entity(
                morph=ezsim.morphs.Sphere(
                    radius=radius,
                    pos=pos,
                    fixed=True  # 设置为固定，不参与物理仿真
                ),
                surface=ezsim.surfaces.Rough(
                    color=color
                )
            )
        elif obstacle_type == 'box':
            scene.add_entity(
                morph=ezsim.morphs.Box(
                    size=size,
                    pos=pos,
                    fixed=True  # 设置为固定，不参与物理仿真
                ),
                surface=ezsim.surfaces.Rough(
                    color=color
                )
            )
        elif obstacle_type == 'cylinder':
            radius = size[0] / 2
            height = size[2]
            scene.add_entity(
                morph=ezsim.morphs.Cylinder(
                    radius=radius,
                    height=height,
                    pos=pos,
                    fixed=True  # 设置为固定，不参与物理仿真
                ),
                surface=ezsim.surfaces.Rough(
                    color=color
                )
            )

def generate_adaptive_obstacles(floor_x: float, floor_y: float, roof_h: float,
                              drone_size: float, difficulty: str = 'medium',
                              occupied_spaces: List[Dict] = None) -> List[Dict]:
    """根据无人机尺寸和难度自适应生成障碍物分布，排除已占用的空间"""
    obstacles = []
    
    if occupied_spaces is None:
        occupied_spaces = []
    
    # 根据难度设置参数
    if difficulty == 'easy':
        obstacle_density = 0.3
        min_clearance = drone_size * 3.0
        size_variation = 0.5
    elif difficulty == 'medium':
        obstacle_density = 0.5
        min_clearance = drone_size * 2.0
        size_variation = 0.7
    else:  # hard
        obstacle_density = 0.7
        min_clearance = drone_size * 1.5
        size_variation = 0.9
    
    # 计算障碍物数量
    area = floor_x * floor_y
    num_obstacles = int(area * obstacle_density / 10)  # 每10平方米的障碍物密度
    
    # 生成障碍物
    max_attempts = num_obstacles * 10
    attempts = 0
    
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        
        # 随机选择障碍物类型
        obstacle_type = random.choice(['sphere', 'box', 'cylinder'])
        
        # 随机位置（避开边界）
        margin = 2.0
        x = random.uniform(-floor_x/2 + margin, floor_x/2 - margin)
        y = random.uniform(-floor_y/2 + margin, floor_y/2 - margin)
        z = random.uniform(0.5, roof_h - 0.5)
        
        # 根据类型生成尺寸
        if obstacle_type == 'sphere':
            base_radius = drone_size * (0.5 + 0.5 * size_variation)
            radius = random.uniform(base_radius * 0.8, base_radius * 1.2)
            size = (2*radius, 2*radius, 2*radius)
        elif obstacle_type == 'box':
            base_size = drone_size * (0.8 + 0.7 * size_variation)
            w = random.uniform(base_size * 0.5, base_size * 1.5)
            h = random.uniform(base_size * 0.5, base_size * 1.5)
            d = random.uniform(base_size * 0.5, base_size * 1.5)
            size = (w, h, d)
        else:  # cylinder
            base_radius = drone_size * (0.3 + 0.4 * size_variation)
            radius = random.uniform(base_radius * 0.8, base_radius * 1.2)
            height = random.uniform(drone_size, roof_h * 0.8)
            size = (2*radius, 2*radius, height)
        
        # 检查是否与现有障碍物或已占用空间太近
        pos = (x, y, z)
        valid_position = True
        
        # 检查与现有障碍物的距离
        for existing in obstacles:
            existing_pos = existing['pos']
            existing_size = existing['size']
            
            distance = np.sqrt((x - existing_pos[0])**2 + 
                             (y - existing_pos[1])**2 + 
                             (z - existing_pos[2])**2)
            min_required_distance = (max(size) + max(existing_size))/2 + min_clearance
            
            if distance < min_required_distance:
                valid_position = False
                break
        
        # 检查与已占用空间的距离（门等）
        if valid_position:
            for occupied in occupied_spaces:
                occupied_pos = occupied['pos']
                occupied_size = occupied['size']
                
                distance = np.sqrt((x - occupied_pos[0])**2 + 
                                 (y - occupied_pos[1])**2 + 
                                 (z - occupied_pos[2])**2)
                
                min_required_distance = (max(size) + max(occupied_size))/2 + min_clearance
                
                if distance < min_required_distance:
                    valid_position = False
                    break
        
        if valid_position:
            obstacles.append({
                'type': obstacle_type,
                'pos': pos,
                'size': size
            })
    
    print(f"Generated {len(obstacles)} obstacles with difficulty '{difficulty}', avoiding {len(occupied_spaces)} occupied spaces")
    return obstacles

def build_all(scene: ezsim.Scene, scene_config: dict):
    """构建完整场景，按顺序：基础结构→门→障碍物"""
    # 获取场景尺寸
    extent = scene_config['scene_extent']
    floor_x, floor_y, roof_h = extent['x'], extent['y'], extent['z']
    
    # 1. 构建基础结构
    build_floor(scene, floor_x, floor_y)
    build_roof(scene, floor_x, floor_y, roof_h)
    build_walls(scene, floor_x, floor_y, roof_h)
    
    # 2. 构建门系统并获取占用空间
    gates_config = scene_config.get('gates', {})
    occupied_spaces = []
    
    if gates_config:
        occupied_spaces = build_gate(scene, gates_config, roof_h)
        print(f"Built {len(occupied_spaces)} gates")
    
    # 3. 生成自适应障碍物（排除门的占用空间）
    drone_size = scene_config.get('drone_size', 0.5)
    difficulty = scene_config.get('difficulty', 'medium')
    
    obstacles_config = generate_adaptive_obstacles(
        floor_x, floor_y, roof_h, drone_size, difficulty, occupied_spaces
    )
    
    # 4. 构建障碍物
    build_obstacles(scene, obstacles_config)
    
    # 5. 构建照明系统
    lighting_config = scene_config.get('lighting', {'grid_x': 15, 'grid_y': 6})
    grid_x = lighting_config.get('grid_x', 15)
    grid_y = lighting_config.get('grid_y', 6)
    build_light(scene, floor_x, floor_y, roof_h, grid_x, grid_y)
    
    return floor_x, floor_y, roof_h


def load_all(yaml_file: str) -> dict:
    """从YAML文件加载配置"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # 返回默认配置
        return {
            'scene_extent': {'x': 20, 'y': 15, 'z': 8},
            'drone_size': 0.5,
            'difficulty': 'medium',
            'lighting': {
                'grid_x': 4,  # X方向灯的数量
                'grid_y': 3   # Y方向灯的数量
            },
            'gates': {
                'gate_1': {
                    'type': 'square',
                    'pos': [5, 0, 3],
                    'size': [3, 3],
                    'direction': 'y',
                    'clearance': 2.5
                },
                'gate_2': {
                    'type': 'circle',
                    'pos': [-5, 3, 4],
                    'size': [2],
                    'direction': 'x',
                    'clearance': 2.0
                }
            }
        } 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-b", "--n_envs", type=int, default=3)
    parser.add_argument("-s", "--n_steps", type=int, default=2)
    parser.add_argument("-r", "--render_all_cameras", action="store_true", default=False)
    parser.add_argument("-o", "--output_dir", type=str, default="img_output/test")
    parser.add_argument("-u", "--use_rasterizer", action="store_true", default=False)
    # video recording options
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--w", type=int, default=640, help="Camera width")
    parser.add_argument("--h", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_len", type=int, default=5, help="Video length in seconds")
    args = parser.parse_args()

    ##########################################
    # backend init
    ##########################################
    ezsim.init(
        seed=4320,
        backend=ezsim.cpu if args.cpu else ezsim.gpu,
        precision="32",
        eps=1e-12,
        log_time=False
    )

    ##########################################
    # scene init
    ##########################################
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(dt=args.dt),
        rigid_options=ezsim.options.RigidOptions(
            box_box_detection=False,
            max_collision_pairs=1000,
            use_gjk_collision=True,
            enable_mujoco_compatibility=False,
        ),
        renderer=ezsim.options.renderers.BatchRenderer(
            use_rasterizer=args.use_rasterizer,
        ),
        vis_options=ezsim.options.VisOptions(show_world_frame=True),
        show_viewer=False,
    )

    ##########################################
    # entity group init/build
    ##########################################
    # 加载配置或使用默认配置
    scene_config = load_all("indoor_config.yaml")  # 如果文件不存在会使用默认配置
    
    # 构建完整场景
    floor_x, floor_y, roof_h = build_all(scene, scene_config)


    ##########################################
    # add cam/recoder/
    # build scene
    # init FrameImageExporter
    ##########################################
    cam_0 = scene.add_camera(
        res=(args.w,args.h),
        pos=(-floor_x//2,0,roof_h//2),
        lookat=(0,0,roof_h//2),
        fov=100,
        GUI=args.vis
    )
    scene.add_light(
        pos=[0.0, 0.0, roof_h-0.5],
        dir=[0.0, 0.0, -1.0],
        directional=0,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.add_light(
        pos=[floor_x/2, -floor_y/2, roof_h-0.5],
        dir=[-1, 1, -1],
        directional=0,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5,
    )

    scene.build(n_envs=args.n_envs)

    exporter = FrameImageExporter(args.output_dir)

    ##########################################
    # render loop
    ##########################################
    for i in range(args.n_steps):
        scene.step()
        rgba, depth, normal_color, seg_color = \
            scene.render_all_cameras(rgb=True, depth=True, normal=True,segmentation=True)
        exporter.export_frame_all_cameras(i, rgb=rgba, depth=depth,normal=normal_color, segmentation=seg_color) 
    


if __name__ == "__main__":
    main()
