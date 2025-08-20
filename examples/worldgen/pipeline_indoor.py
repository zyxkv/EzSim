from typing import Dict, List, Tuple, Optional, Union, Any, Literal
from pydantic import BaseModel, Field
import numpy as np
import torch 
from numpy.typing import NDArray
import argparse
import yaml
import random


import ezsim
from ezsim.sensors import SensorDataRecorder, VideoFileWriter
from ezsim.utils.image_exporter import FrameImageExporter

#######################
# Room
#######################
class Room(BaseModel):
    """室内房间配置"""
    extent: Tuple[float, float, float] = Field(default=(40.0, 10.0, 8.0), description='房间尺寸 (长, 宽, 高)')
    wall_thickness: float = Field(default=0.2, description='墙壁厚度')
    roof_thickness: float = Field(default=0.2, description='屋顶厚度')
    textures: Dict[str, Union[str,Tuple[float,float,float]]] = Field(default_factory=lambda: {
        'floor': "textures/carpet_grayblue_01.png",
        'roof': "textures/stucco_wall_blue.png",
        'left': (0.2, 0.2, 0.9), 
        'right': (0.9, 0.2, 0.2),
        'front': (0.2, 0.9, 0.2),
        'back': (0.9, 0.2, 0.9)
    }, description='地板/屋顶/墙面纹理或颜色配置，支持RGB颜色元组或纹理路径')

def build_room(scene: ezsim.Scene, room_config: Room, SEQ_CODE:int=0):
    assert SEQ_CODE & 0b0001, f'[step1: init scene ] is not ready'
    ex, ey, roof_h = room_config.extent
    # 地板
    scene.add_entity(
        morph=ezsim.morphs.Plane(fixed=True, contype=0x0001, conaffinity=0x0001),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['floor'])) 
            if isinstance(room_config.textures['floor'], str) else ezsim.surfaces.Rough(color=room_config.textures['floor']) 
    )      
    # 屋顶
    scene.add_entity(
        morph=ezsim.morphs.Mesh(file="meshes/drone_racing/box.obj", pos=(0, 0, roof_h + room_config.roof_thickness/8), scale=(ex + 2*room_config.wall_thickness, ey + 2*room_config.wall_thickness, room_config.roof_thickness), fixed=True, contype=0x0002, conaffinity=0x0002),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['roof'])) 
            if isinstance(room_config.textures['roof'], str) else ezsim.surfaces.Rough(color=room_config.textures['roof']) 
    )
    # 墙壁
    # 左墙
    scene.add_entity(
        morph=ezsim.morphs.Plane(pos=(0, ey / 2, roof_h / 2), normal=(0, -1, 0), fixed=True, contype=0x0004, conaffinity=0x0004),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['left'])) 
            if isinstance(room_config.textures['left'], str) else ezsim.surfaces.Rough(color=room_config.textures['left'])
    )
    # 右墙
    scene.add_entity(
        morph=ezsim.morphs.Plane(pos=(0, -ey / 2, roof_h / 2), normal=(0, 1, 0), fixed=True, contype=0x0004, conaffinity=0x0004),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['right'])) 
            if isinstance(room_config.textures['right'], str) else ezsim.surfaces.Rough(color=room_config.textures['right']) 
    )
    # 前墙
    scene.add_entity(
        morph=ezsim.morphs.Plane(pos=(ex / 2, 0, roof_h / 2), normal=(-1, 0, 0), fixed=True, contype=0x0008, conaffinity=0x0008),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['front'])) 
            if isinstance(room_config.textures['front'], str) else ezsim.surfaces.Rough(color=room_config.textures['front']) 
    )
    # 后墙
    scene.add_entity(
        morph=ezsim.morphs.Plane(pos=(-ex / 2, 0, roof_h / 2), normal=(1, 0, 0), fixed=True, contype=0x0008, conaffinity=0x0008),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=room_config.textures['back'])) 
            if isinstance(room_config.textures['back'], str) else ezsim.surfaces.Rough(color=room_config.textures['back']) 
    )
    return 0b0010, scene

#######################
# Lights
#######################
class Lights(BaseModel):
    """室内照明配置"""
    ceil: Dict[str,Any] = Field(default_factory=lambda: {
        'color':[1.0,1.0,1.0], 'grid':[15,6], 'margin': [5.0, 2.0],'height': 7.8, 
        'directional':0, 'cutoff_deg': 45, 'attenuation': 0.1,'intensity': 0.05,
    })
    wall: Dict[str, Any] = Field(default_factory=lambda: {
        'color':[1.0,1.0,1.0], 'margin':[0.5,0.5], 'height': 6.0, 
        'directional':0, 'cutoff_deg': 90, 'attenuation': 0.001,'intensity': 0.08,
    })
    ambient: Dict[str, Any] = Field(default_factory=lambda: {
        'color':[1.0,1.0,1.0], 'num': 4, 'height': 7.8, 
        'directional':0, 'cutoff_deg': 90, 'attenuation': 0.1,'intensity': 0.125,
    })

def build_light(scene: ezsim.Scene, light_config: Lights, SEQ_CODE:int=0, room_size:Tuple[float, float, float]=(40,10,8)):
    assert SEQ_CODE & 0b0010, f'[step 2: build room] is not ready'
    ex, ey, room_h = room_size
    mex, mey = light_config.ceil['margin']
    avax, avay = ex-2*mex, ey-2*mey
    if avax <= 0 or avay <= 0:
        raise ezsim.logger.warning(f"Warning: Room too small for lighting with margins. Room: {ex}x{ey}, Required: {2*mex}x{2*mey}")
    # 计算天花板灯之间的间距和起始位置
    grid_x, grid_y = light_config.ceil['grid']
    ceil_lpos_x = [-avax/2 + (i* avax/(grid_x - 1)) for i in range(grid_x)] if grid_x > 1 else [0]
    ceil_lpos_y = [-avay/2 + (j* avay/(grid_y - 1)) for j in range(grid_y)] if grid_y > 1 else [0]

    assert light_config.ceil['height'] < room_h, f"Ceiling light height {light_config.ceil['height']} must be less than room height {room_h}"
    for clpx in ceil_lpos_x:
        for clpy in ceil_lpos_y:
            pos = (clpx, clpy, light_config.ceil['height'])
            scene.add_light(
                pos=pos,
                dir=[0.0, 0.0, -1.0],
                directional=light_config.ceil['directional'],
                castshadow=1,
                cutoff=light_config.ceil['cutoff_deg'],
                intensity=light_config.ceil['intensity'],
                attenuation=light_config.ceil['attenuation']
            )
    # 墙面补充照明 - 在天花板灯的间隔处添加补充光源以消除光影图案
    mwx,mwy = light_config.wall['margin']
    wlh = light_config.wall['height']
    assert wlh < room_h, f"Wall light height {wlh} must be less than room height {room_h}"
    wall_lpos_x = [(ceil_lpos_x[i + 1] + ceil_lpos_x[i]) / 2 for i in range(len(ceil_lpos_x) - 1)]
    for wlpx in wall_lpos_x:
        # 左墙补充照明
        scene.add_light(
            pos=[wlpx, ey/2 - mwy , wlh],
            dir=[0.0,-1.0,0.0],
            directional=light_config.wall['directional'],
            castshadow=0,
            cutoff=light_config.wall['cutoff_deg'],
            intensity=light_config.wall['intensity'],
            attenuation=light_config.wall['attenuation']
        )
        # 右墙补充照明
        scene.add_light(
            pos=[wlpx, -ey/2 + mwy , wlh],
            dir=[0.0, 1.0, 0.0],
            directional=light_config.wall['directional'],
            castshadow=0,
            cutoff=light_config.wall['cutoff_deg'],
            intensity=light_config.wall['intensity'],
            attenuation=light_config.wall['attenuation']
        )
    wall_lpos_y = [(ceil_lpos_y[i + 1] + ceil_lpos_y[i]) / 2 for i in range(len(ceil_lpos_y) - 1)]
    for wlpy in wall_lpos_y:
        # 前墙补充照明
        scene.add_light(
            pos=[ex/2 - mwx, wlpy, wlh],
            dir=[-1.0, 0.0, 0.0],
            directional=light_config.wall['directional'],
            castshadow=0,
            cutoff=light_config.wall['cutoff_deg'],
            intensity=light_config.wall['intensity'],
            attenuation=light_config.wall['attenuation']
        )
        # 后墙补充照明
        scene.add_light(
            pos=[-ex/2 + mwx, wlpy, wlh],
            dir=[1.0, 0.0, 0.0],
            directional=light_config.wall['directional'],
            castshadow=0,
            cutoff=light_config.wall['cutoff_deg'],
            intensity=light_config.wall['intensity'],
            attenuation=light_config.wall['attenuation']
        )
    # 添加环境光以进一步平滑光照
    amlight_num = light_config.ambient['num']
    assert light_config.ambient['height'] < room_h, f"Ambient light height {light_config.ambient['height']} must be less than room height {room_h}"
    for i in range(amlight_num):
        angle = i * 2 * np.pi / amlight_num
        radius = min(ex, ey) * 0.3
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)
        
        scene.add_light(
            pos=[x_pos, y_pos, light_config.ambient['height']],
            dir=[0.0, 0.0, -1.0],
            directional=light_config.ambient['directional'],
            castshadow=0,  # 环境光不产生阴影
            cutoff=light_config.ambient['cutoff_deg'],
            attenuation= light_config.ambient['attenuation'],
            intensity= light_config.ambient['intensity'],
        )
    
    # 统计信息
    total_wall_lights = 2 * len(wall_lpos_x) + 2 * len(wall_lpos_y)
    total_ceiling_lights = grid_x * grid_y
    
    ezsim.logger.info(f"Built lighting system:")
    ezsim.logger.info(f"  - {total_ceiling_lights} ceiling lights ({grid_x}x{grid_y} grid) at {light_config.ceil['height']:.1f}m")
    ezsim.logger.info(f"  - {total_wall_lights} wall lights ({len(wall_lpos_y)} front/back + {len(wall_lpos_x)} left/right each) at {light_config.wall['height']:.1f}m")
    ezsim.logger.info(f"  - {amlight_num} ambient lights at {light_config.ambient['height']:.1f}m")
    ezsim.logger.info(f"  - Wall lights positioned between ceiling light intervals to eliminate shadow patterns") 
    SEQ_CODE = 0b0100
    return SEQ_CODE, scene

#######################
# GeomObj
#######################
class GeomObj(BaseModel):
    """几何体配置"""
    pos: Tuple[float, float, float] = Field(default=(0, 0, 0), description='几何体位置 (x, y, z)')
    size: Tuple[float, float, float] = Field(default=(0.3, 0.3, 0.3), description='几何体尺寸 (宽, 高, 深)')
    euler: Tuple[float, float, float] = Field(default=(0, 0, 0), description='几何体欧拉角 (roll, pitch, yaw)')
    texture: Union[str, Tuple[float,float,float]] = Field(default=(1.0, 0.5, 0.0), description='几何体颜色或纹理路径')
    aabb: Any = Field(default=np.zeros((2, 3)), description='几何体的轴对齐包围盒')

#######################
# Gates
#######################
"""
室内门配置
真实门：
SJTU方形门 外框 1.3m *1.3m 内框 1m*1m 深度 0.05m
圆形红色圈门 目测 1.4m *1.4m
"""

class Gate(GeomObj):
    """门配置"""
    mesh_file: str = Field(description='门的模型文件路径')
    texture = Field(description='门的纹理路径或颜色')
    raw_size: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0), description='门模型原始尺寸 (宽, 高, 深)')
    direction: Literal['x','y','z'] = Field(default='x', description='门的方向')

    def post_place(self, robot_size: Tuple[float, float, float]= (0.3,0.3,0.3), path_ratio:float=3.0):
        """根据机器人尺寸和门的方向计算欧拉角和包围盒"""
        exh = 0.5*self.size[0] + path_ratio*robot_size[0]
        eyh = 0.5*self.size[1] + path_ratio*robot_size[1] 
        ezh = 0.5*self.size[2] + path_ratio*robot_size[2]
        
        if self.direction == 'x':
            self.euler = (90.0, 0.0, -90.0)
            self.aabb = np.array([
                [self.pos[0]-ezh, self.pos[1]-eyh, self.pos[2]-exh],
                [self.pos[0]+ezh, self.pos[1]+eyh, self.pos[2]+exh]
            ])
        elif self.direction == 'y':
            self.euler = (90.0, 0.0, 0.0)
            self.aabb = np.array([
                [self.pos[0]-exh, self.pos[1]-ezh, self.pos[2]-eyh],
                [self.pos[0]+exh, self.pos[1]+ezh, self.pos[2]+eyh]
            ])
        elif self.direction == 'z':
            self.euler = (0.0, 0.0, 0.0)
            self.aabb = np.array([
                [self.pos[0]-exh, self.pos[1]-eyh, self.pos[2]-ezh],
                [self.pos[0]+exh, self.pos[1]+eyh, self.pos[2]+ezh]
            ])
        else:
            ezsim.logger.warning(f"Unsupported door direction: {self.direction}, euler will keep {self.euler}")
    
    @property
    def scale(self)->Tuple[float, float, float]:
        """
        返回仿真器morph.Mesh中的缩放因子
        """
        return self.size[0]/self.raw_size[0], self.size[1]/self.raw_size[1], self.size[2]/self.raw_size[2]

class SquareGate(Gate):
    mesh_file = Field(default="meshes/drone_racing/square_gate.obj", description='方形门的模型文件路径')
    texture = Field(default="textures/square_gate.png", description='方形门的纹理路径')
    raw_size = Field(default=(1.3, 1.3, 0.05), description='门模型原始尺寸 (宽, 高, 深)')
    
class CircleGate(Gate):
    mesh_file = Field(default="meshes/drone_racing/circle_gate.obj", description='圆形门的模型文件路径')
    texture = Field(default=(0.925, 0.015, 0.0), description='圆形门的纹理路径')
    raw_size = Field(default=(1.4, 1.4, 0.1), description='门模型原始尺寸 (宽, 高, 深)')


def build_gates(scene: ezsim.Scene, gates_config: Dict, occ_space:List, SEQ_CODE:int=0, robot_size:Tuple[float, float, float]=(0.3,0.3,0.3), path_ratio:float=3.0):
    assert SEQ_CODE & 0b0100, f'[step3: build light] is not ready'
    for gate_name, gate_dict in gates_config.items():
        if gate_name.startswith('square'):
            gate_type = 'square' 
        elif gate_name.startswith('circle'):
            gate_type = 'circle'
        else:
            raise ValueError(f"Unsupported gate type: {gate_name}")
        gate = SquareGate(gate_dict) if gate_name.startswith('square') else CircleGate(gate_dict)
        gate.post_place(robot_size,path_ratio)
        scene.add_entity(
            morph=ezsim.morphs.Mesh(file=gate.mesh_file, pos=gate.pos, euler=gate.euler, scale=gate.scale, fixed=True),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=gate.texture)) 
                    if isinstance(gate.texture, str) else ezsim.surfaces.Rough(color=gate.texture)
            )

        occ_space.append({'name': gate_name, 'obj_type': gate_type,'pos':gate.pos,'aabb':gate.aabb})
        ezsim.logger.info(f"Added {gate_type} gate '{gate_name}' at {gate.pos} with size {gate.size}")
    SEQ_CODE = 0b1000
    return SEQ_CODE, scene, occ_space


#######################
# Obstacles
#######################
"""
室内障碍物配置
真实障碍物：
[暂不支持] 儿童EPP泡沫大乐高块 0.3m x 0.15m x 0.15m, 连接处高0.035m

虚拟障碍物：
柱子(): 
"""

def prealloc_obstacles(scene:ezsim.Scene, obstacles_config:Dict, occ_space:List, SEQ_CODE:int=0):
    assert SEQ_CODE & 0b1000, f'[step4: build gates] is not ready'
    prealloc_obs_dict = dict()
    roof_prob, roof_height, rf_expsize = obstacles_config['roof']['prob'], obstacles_config['roof']['height'], obstacles_config['roof']['size']
    
    # pillar 预生成
    N_p = obstacles_config['pillar']['num'] 
    P_lb_p, P_ub_p = obstacles_config['pillar']['pos'][0], obstacles_config['pillar']['pos'][1]
    S_lb_p, S_ub_p = obstacles_config['pillar']['size'][0], obstacles_config['pillar']['size'][1]
    Inc_p = obstacles_config['pillar']['incidence']
    # 根据P_lb_p, P_ub_p S_lb_p, S_ub_p 生成柱子的随机位置和尺寸，并随机选择是cylinder还是box 柱子
    # Inc_p 是柱子在YoZ平面的最大倾斜角，发生倾斜的概率为 0.1， 如果倾斜，则先在z方向延长1m高度，然后pos向下移动1m。然后再进行柱子的旋转


    # # box 预生成
    # N_b = obstacles_config['box']['num'] 
    # P_lb_b, P_ub_b = obstacles_config['box']['pos'][0], obstacles_config['box']['pos'][1]
    # S_lb_b, S_ub_b = obstacles_config['box']['size'][0], obstacles_config['box']['size'][1]
    


    # off_roof_b = torch.rand((N_b,)) < roof_prob
    # # sphere 预生成
    # N_s = obstacles_config['sphere']['num'] 
    # P_lb_s, P_ub_s = obstacles_config['sphere']['pos'][0], obstacles_config['sphere']['pos'][1]
    # S_lb_s, S_ub_s = obstacles_config['sphere']['size'][0], obstacles_config['sphere']['size'][1]
    # off_roof_s = torch.rand((N_s,)) < roof_prob
    # # bump 预生成
    # N_bu = obstacles_config['bump']['num']
    # P_lb_u, P_ub_u = obstacles_config['bump']['pos'][0], obstacles_config['bump']['pos'][1]
    # S_lb_u, S_ub_u = obstacles_config['bump']['size'][0], obstacles_config['bump']['size'][1]

    # # arrest 预生成
    # N_a = obstacles_config['arrest']['num']
    # P_lb_a, P_ub_a = obstacles_config['arrest']['pos'][0], obstacles_config['arrest']['pos'][1]
    # S_lb_a, S_ub_a = obstacles_config['arrest']['size'][0], obstacles_config['arrest']['size'][1]
    # Inc_a = obstacles_config['arrest']['incidence']

    SEQ_CODE = 0b10000
    return SEQ_CODE, prealloc_obs_dict, occ_space

def build_obstacles(scene:ezsim.Scene, prealloc_obs_dict:Dict,SEQ_CODE:int=0):
    assert SEQ_CODE & 0b10000, f'[step5: prealloc_obstacles] is not ready'
    # add entity based on prealloc_obs_dict:

    return scene 









def main():


if 