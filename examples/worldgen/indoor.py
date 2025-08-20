import torch 
import math 
import numpy as np
from numpy.typing import NDArray
import argparse
import yaml
import random
from typing import Dict, List, Tuple, Optional, Union, Any

import ezsim
from ezsim.sensors import SensorDataRecorder, VideoFileWriter
from ezsim.utils.image_exporter import FrameImageExporter

from typing import Literal
from pydantic import BaseModel, Field


#######################
# Global Scene
#######################
global SCENE 
SCENE = None
global SCENE_STATUS 
SCENE_STATUS = -1

###############################################
# BackEnd/Scene init (only run once)
###############################################
def init_scene(args: argparse.Namespace) -> Tuple[int,Optional[ezsim.Scene]]:
    global SCENE, SCENE_STATUS
    if SCENE_STATUS < 0:
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
        SCENE = ezsim.Scene(
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
        SCENE_STATUS = 0
        return SCENE_STATUS, SCENE # Scene initialized
    return SCENE_STATUS, SCENE  # Scene already exists


#######################
# Room
#######################
class Room(BaseModel):
    """室内房间配置"""
    extent: Tuple[float, float, float] = Field(default=(40.0, 10.0, 8.0), description='房间尺寸 (长, 宽, 高)')
    textures: Dict[str, Union[str,Tuple[float,float,float]]] = Field(default_factory=lambda: {
        'floor': "textures/carpet_grayblue_01.png",
        'roof': "textures/stucco_wall_blue.png",
        'left': (0.2, 0.2, 0.9), 
        'right': (0.9, 0.2, 0.2),
        'front': (0.2, 0.9, 0.2),
        'back': (0.9, 0.2, 0.9)
    }, description='地板/屋顶/墙面纹理或颜色配置，支持RGB颜色元组或纹理路径')

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

#######################
# Gates
#######################
class GeomObj(BaseModel):
    """几何体配置"""
    pos: Tuple[float, float, float] = Field(default=(0, 0, 0), description='几何体位置 (x, y, z)')
    size: Tuple[float, float, float] = Field(default=(0.3, 0.3, 0.3), description='几何体尺寸 (宽, 高, 深)')
    euler: Tuple[float, float, float] = Field(default=(0, 0, 0), description='几何体欧拉角 (roll, pitch, yaw)')
    texture: Union[str, Tuple[float,float,float]] = Field(default=(1.0, 0.5, 0.0), description='几何体颜色或纹理路径')
    aabb: Any = Field(default=np.zeros((2, 3)), description='几何体的轴对齐包围盒')

class SquareGate(GeomObj):
    texture = Field(default="textures/sjtu_square_gate.png", description='门的纹理路径')
    direction: Literal['x','y','z'] = Field(default='x', description='门的方向')
    mesh_file: str = Field(default='meshes/drone_racing/square_gate.obj', description='门的模型文件路径')
    
    def post_euler(self, robot_size: Tuple[float,float,float] = (0.3,0.3,0.3), path_ratio:float=3.0):
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
        return self.size[0] / 1.3, self.size[1] / 1.3, self.size[2] / 0.05

class CircleGate(GeomObj):
    texture = Field(default=(0.925, 0.015, 0.0), description='门的颜色或纹理路径')
    direction: Literal['x','y','z'] = Field(default='x', description='门的方向')
    mesh_file: str = Field(default='meshes/drone_racing/circle_gate.obj', description='门的模型文件路径')


    def post_euler(self, robot_size: Tuple[float,float,float] = (0.3,0.3,0.3), path_ratio:float=3.0):
        """根据机器人尺寸和门的方向计算欧拉角和包围盒"""
        exh = self.size[0] + path_ratio*robot_size[0]
        eyh = self.size[1] + path_ratio*robot_size[1]
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
        return self.size[0] / 0.7, self.size[1] / 0.7, self.size[2] / 0.1
    
def add_to_scene(geom: GeomObj, scene:ezsim.Scene, occupied_spaces:List, geom_name: str='geom_obj',geom_type:str='static'):
    scene.add_entity(morph=ezsim.morphs.Mesh(
            file=geom.mesh_file,
            pos=geom.pos,
            euler=geom.euler,
            scale=geom.scale,
            fixed=True
        ),
        surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=geom.texture))
            if isinstance(geom.texture, str) else ezsim.surfaces.Rough(color=geom.texture)
    )
    occupied_spaces.append({'name': geom_name,'type': geom_type,'pos': geom.pos,'aabb': geom.aabb})


def build_gate(scene:ezsim.Scene, occupied_space:List, gate_config:Dict, robot_size: Tuple[float,float,float] = (0.3,0.3,0.3)):
    path_ratio = gate_config.get('path_ratio', 3.0)
    sgate_config = gate_config.get('square',{})
    if sgate_config and len(sgate_config['pos']) > 0:
        assert len(sgate_config['size']) == len(sgate_config['pos'])
        assert len(sgate_config['direction']) == len(sgate_config['pos'])
        for i in range(len(sgate_config['pos'])):
            gate = SquareGate(
                pos=sgate_config['pos'][i],
                size=sgate_config['size'][i],
                direction=sgate_config['direction'][i],
                path_ratio=path_ratio
            )
            occupied_space.append(gate)

    cgate_config = gate_config.get('circle',{})


class SquareGate(BaseModel):
    """方形门配置[obj out_size 1.3x1.3 inner_frame 1.0x1.0 thickness 0.05]"""
    pos: Tuple[float, float, float] = Field(default=(0, 0, 1.5), description='门的位置 (x, y, z)')
    size: Tuple[float, float, float] = Field(default=(1.3, 1.3, 0.05), description='门的尺寸 (宽, 高, 深)')
    direction: Literal['x','y','z'] = Field(default='x', description='门的方向')
    euler: Tuple[float, float, float] = Field(default=(0, 0, 0), description='门的欧拉角 (roll, pitch, yaw)')
    path_ratio: float = Field(default=2.0, description='门周围预留可通过路径倍数')
    texture: Union[str, Tuple[float,float,float]] = Field(default="textures/sjtu_square_gate.png", description='门的纹理')
    aabb:Any = Field(default=np.zeros((2, 3)), description='门的轴对齐包围盒')
    
    def post_euler(self, robot_size: Tuple[float,float,float] = (0.3,0.3,0.3)):
        """根据机器人尺寸和门的方向计算欧拉角和包围盒"""
        exh = 0.5*self.size[0] + self.path_ratio*robot_size[0]
        eyh = 0.5*self.size[1] + self.path_ratio*robot_size[1] 
        ezh = 0.5*self.size[2] + self.path_ratio*robot_size[2]
        
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
        return self.size[0] / 1.3, self.size[1] / 1.3, self.size[2] / 0.05
    
class CircleGate(BaseModel):
    """圆形门配置[obj out_radius 0.7 inner_radius 0.5 thickness 0.1]"""
    pos: Tuple[float, float, float] = Field(default=(0, 0, 1.7), description='门的位置 (x, y, z)')
    size: Tuple[float, float, float] = Field(default=(0.7, 0.7, 0.1), description='门的尺寸 (宽, 高, 深)')
    direction: Literal['x','y','z'] = Field(default='x', description='门的方向')
    euler: Tuple[float, float, float] = Field(default=(0, 0, 0), description='门的欧拉角 (roll, pitch, yaw)')
    path_ratio: float = Field(default=2.0, description='门周围预留可通过路径倍数')
    texture: Union[str, Tuple[float,float,float]] = Field(default=(0.925,0.015,0.0), description='门的纹理')
    aabb:Any = Field(default=np.zeros((2, 3)), description='门的轴对齐包围盒')

    def post_euler(self, robot_size: Tuple[float,float,float] = (0.3,0.3,0.3)):
        """根据机器人尺寸和门的方向计算欧拉角和包围盒"""
        exh = self.size[0] + self.path_ratio*robot_size[0]
        eyh = self.size[1] + self.path_ratio*robot_size[1]
        ezh = 0.5*self.size[2] + self.path_ratio*robot_size[2]
        
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
        return self.size[0] / 0.7, self.size[1] / 0.7, self.size[2] / 0.1

#######################
# Cam
#######################
class Cam(BaseModel):
    pos: Tuple[float, float, float] = Field(default=(-20, 0, 5), description='相机位置 (x, y, z)')
    lookat: Tuple[float, float, float] = Field(default=(0, 0, 3), description='相机朝向目标点 (x, y, z)')
    res: Tuple[int,int] = Field(default=(1280, 960), description='相机分辨率 (宽, 高)')
    fov: float = Field(default=80, description='相机视场角 (度)')
    # fps: int = Field(default=30, description='相机帧率') # 目前不支持
    gui: bool = Field(default=False, description='是否启用GUI')


#######################
# Obstacles
#######################
"""
室内障碍物配置
真实情况：儿童EPP泡沫， 0.15x0.15x0.3, 连接处高0.035
"""

class Pillar(Obstacle):

class Voxel(Obstacle):

class Ball(Obstacle):

class Bump(Obstacle):

class Arrest(Obstacle):



class Obstacles(BaseModel):
    """室内障碍物配置"""
    types: List[str] = Field(default_factory=lambda: ['sphere', 'box', 'cylinder', 
                                                      'triangular_pyramid', 'square_pyramid', 
                                                      'square_frustum', 'cone_frustum'])
    count: int = Field(default=80, description='房间内的障碍物数量')                     
    density: float = Field(default=0.2, description='障碍物/平方米')  # 
    path_ratio: float = Field(default=2.0, description='障碍物之间预留可通过路径倍数') 
    size_variation: Tuple[float,float,float] = Field(default=(0.5, 0.5, 0.125), description= '障碍物本身尺寸变化范围')


#######################
# Indoor Scene
#######################
class IndoorScene:
    """室内场景配置"""
    scene: Optional[ezsim.Scene] = None
    cameras: Dict[str, Cam] = {}
    
    def __init__(self, **data):
        # 先初始化为默认配置，避免属性未定义的问题
        self.default_init()
        # 设置scene为全局SCENE
        global SCENE
        self.scene = SCENE
        # 如果有提供yaml_path，则从YAML文件加载配置
        if 'yaml_path' in data:
            self.load_yaml(yaml_path=data['yaml_path'])

    def default_init(self):
        """返回默认配置"""
        self.room = Room()
        self.lights = Lights()
        self.robot_size: Tuple[float, float, float] = (0.3, 0.3, 0.3)
        self.gates = dict(square={}, circle={})
        self.obstacles = Obstacles()
        self.cameras = dict()  # 相机配置字典
        self.occupied_spaces: List[Dict] = []  # 用于记录已占用的空间
        self._debug_occupied_spaces = False
        #

    def load_yaml(self, yaml_path):
        """从YAML文件加载配置"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 加载场景尺寸配置
            scene_extent_config = config.get('scene_extent', {})
            self.room = Room(extent=(
                scene_extent_config.get('x', 40.0),
                scene_extent_config.get('y', 10.0), 
                scene_extent_config.get('z', 8.0)
            ))
            
            # 加载光照配置
            lighting_config = config.get('lighting', {})
            ceil_config = lighting_config.get('ceil', {})
            wall_config = lighting_config.get('wall', {})
            ambient_config = lighting_config.get('ambient', {})
            
            self.lights = Lights(
                ceil={
                    'color': ceil_config.get('color', [1.0, 1.0, 1.0]),
                    'grid': [ceil_config.get('grid_x', 15), ceil_config.get('grid_y', 6)],
                    'margin': [ceil_config.get('margin_x', 5.0), ceil_config.get('margin_y', 2.0)],
                    'height': ceil_config.get('height', 7.8),
                    'directional': ceil_config.get('directional', 0),
                    'cutoff_deg': ceil_config.get('cutoff_deg', 45),
                    'attenuation': ceil_config.get('attenuation', 0.1),
                    'intensity': ceil_config.get('intensity', 0.05),
                },
                wall={
                    'color': wall_config.get('color', [1.0, 1.0, 1.0]),
                    'margin': [wall_config.get('margin_x', 0.5), wall_config.get('margin_y', 0.5)],
                    'height': wall_config.get('height', 6.0),
                    'directional': wall_config.get('directional', 0),
                    'cutoff_deg': wall_config.get('cutoff_deg', 90),
                    'attenuation': wall_config.get('attenuation', 0.001),
                    'intensity': wall_config.get('intensity', 0.08),
                },
                ambient={
                    'color': ambient_config.get('color', [1.0, 1.0, 1.0]),
                    'num': ambient_config.get('num', 4),
                    'height': ambient_config.get('height', 7.8),
                    'directional': ambient_config.get('directional', 0),
                    'cutoff_deg': ambient_config.get('cutoff_deg', 90),
                    'attenuation': ambient_config.get('attenuation', 0.1),
                    'intensity': ambient_config.get('intensity', 0.125),
                }
            )
            
            # 加载门配置
            gates_config = config.get('gates', {})
            self.gates = {
                'square': gates_config.get('square', {}),
                'circle': gates_config.get('circle', {})
            }
            
            # 加载机器人尺寸配置
            robot_config = config.get('robot', {})
            rbt_size = robot_config.get('size', [0.3, 0.3, 0.3])
            rbt_vts = robot_config.get('vts', 1.0)
            dp_fwd_min,dp_fwd_max = robot_config.get('dp_fwd', [0.75, 5.75])
            self.robot_size = tuple(rbt_size)
            self.robot_vts = rbt_vts
            self.robot_dp_fwd_min = dp_fwd_min * rbt_vts
            self.robot_dp_fwd_max = dp_fwd_max * rbt_vts

            # 加载障碍物配置
            obstacles_config = config.get('obstacles', {})
            self.obstacles = Obstacles(
                types=obstacles_config.get('types', ['sphere', 'box', 'cylinder']),
                count=obstacles_config.get('count', 50),
                density=obstacles_config.get('density', 0.2),
                path_ratio=obstacles_config.get('path_ratio', 2.0),
                size_variation=tuple(obstacles_config.get('size_variation', [0.5, 0.5, 0.125]))
            )
            
            # 加载相机配置
            camera_config = config.get('camera', {})
            self.cameras = {}
            for cam_name, cam_cfg in camera_config.items():
                self.cameras[cam_name] = Cam(
                    pos=tuple(cam_cfg.get('pos', [-20, 0, 5])),
                    lookat=tuple(cam_cfg.get('lookat', [0, 0, 3])),
                    res=tuple(cam_cfg.get('res', [1280, 960])),
                    fov=cam_cfg.get('fov', 80),
                    gui=cam_cfg.get('gui', False)
                )
            
            print(f"Successfully loaded configuration from {yaml_path}")
            
        except FileNotFoundError:
            print(f"Warning: YAML file {yaml_path} not found. Using default configuration.")
            return self.default_init()   

    def build(self, n_envs:int=1):
        self.scene.build(n_envs=n_envs)
        self.exporter = FrameImageExporter('img_output/indoor_dojo')
        
    def run(self, steps:int=3, w_rgb:bool=True, w_depth:bool=True, w_normal:bool=False, w_seg:bool=False):
        ##########################################
        # render loop
        ##########################################
        if self._debug_occupied_spaces:
            self.recorder.start_recording()
        try:
            for i in range(steps):
                # 在场景构建后添加调试可视化
                if self._debug_occupied_spaces:
                    self._debug_objects = self.draw_debug_occupied_spaces()
                self.scene.step()
                rgba, depth, normal_color, seg_color = self.scene.render_all_cameras(w_rgb, w_depth, w_normal, w_seg)
                self.exporter.export_frame_all_cameras(i, rgb=rgba, depth=depth,normal=normal_color, segmentation=seg_color)
                if self._debug_occupied_spaces:
                    self.recorder.step()
                    self.scene.clear_debug_objects()  # 清除调试对象以避免内存泄漏
        except KeyboardInterrupt:
            ezsim.logger.info("Simulation interrupted, exiting.")
        finally:
            ezsim.logger.info("Simulation finished.")
            if self._debug_occupied_spaces:
                self.recorder.stop_recording()
            
    # TODO: 需要实现build_obstacles
    def construct(self, debug_occupied_spaces: bool = False):
        # 存储调试标志，在build()中使用
        self._debug_occupied_spaces = debug_occupied_spaces

        # pipeline: room -> lights -> gates -> cameras -> obstacles
        self._build_room()._build_lights()._build_gates()
        _ = self.prealloc_obstacles()
        self.build_obstacles()
        if self._debug_occupied_spaces:
            self.recorder = SensorDataRecorder(step_dt=1e-2)
        self._build_cameras()
        if self._debug_occupied_spaces:
            self.recorder.add_sensor(self.camera_objects[0], VideoFileWriter(filename="video_res/indoor_dojo_debug.mp4"))

        return self
        
    def add_debug_cam(self):
        """添加默认的调试相机（向后兼容的方法）"""
        debug_cam = self.scene.add_camera(
            res=(1280,960),
            pos=(-self.room.extent[0]//2,0,self.room.extent[2]//2),
            lookat=(0,0,self.room.extent[2]//2),
            fov=100,
            GUI=False
        )
        return debug_cam

    def _build_cameras(self):
        """根据配置构建相机系统"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call init_scene(args) first.")
        
        self.camera_objects = []
        for cam_name, cam_config in self.cameras.items():
            camera = self.scene.add_camera(
                res=cam_config.res,
                pos=cam_config.pos,
                lookat=cam_config.lookat,
                fov=cam_config.fov,
                GUI=cam_config.gui
            )
            self.camera_objects.append(camera)
            ezsim.logger.info(f"Added camera '{cam_name}' at {cam_config.pos} looking at {cam_config.lookat}")
        
        # 如果没有配置任何相机，添加默认的调试相机
        if not self.cameras:
            ezsim.logger.info("No cameras configured, adding default debug camera")
            default_cam = self.add_debug_cam()
            self.camera_objects.append(default_cam)
        
        global SCENE_STATUS
        SCENE_STATUS = 4
        return self

    def draw_debug_occupied_spaces(self):
        """绘制occupied_spaces的外包围盒用于调试"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call init_scene(args) first.")
        
        debug_objects = []
        for i, space in enumerate(self.occupied_spaces):
            if 'aabb' in space:
                aabb = space['aabb']
                space_name = space.get('name', f'space_{i}')
                space_type = space.get('type', 'unknown')
                
                # 根据空间类型设置不同的颜色
                if 'gate_square' in space_type:
                    color = (0.0, 1.0, 0.0, 0.8)  # 绿色 - 方形门
                elif 'gate_circle' in space_type:
                    color = (0.0, 0.0, 1.0, 0.8)  # 蓝色 - 圆形门
                elif 'obstacle' in space_type:
                    color = (1.0, 0.5, 0.0, 0.8)  # 橙色 - 障碍物
                else:
                    color = (1.0, 0.0, 1.0, 0.8)  # 紫色 - 未知类型
                
                # 绘制wireframe包围盒
                debug_obj = self.scene.draw_debug_box(
                    bounds=aabb,
                    color=color,
                    wireframe=True,
                    wireframe_radius=0.005
                )
                debug_objects.append(debug_obj)
                
                ezsim.logger.info(f"Drew debug box for '{space_name}' ({space_type})")
                ezsim.logger.info(f"  AABB bounds: {aabb}")
        
        ezsim.logger.info(f"Total debug boxes drawn: {len(debug_objects)}")
        return debug_objects

    def clear_debug_objects(self):
        """清除所有调试对象"""
        if hasattr(self, '_debug_objects') and self._debug_objects:
            for debug_obj in self._debug_objects:
                try:
                    self.scene.clear_debug_object(debug_obj)
                except Exception as e:
                    ezsim.logger.warning(f"Failed to clear debug object: {e}")
            self._debug_objects = []
            ezsim.logger.info("Cleared all debug objects")

    # FINISH: 已完成/基本无需检查 矩形room空间，天花板地面和四面墙
    def _build_room(self):
        """根据配置构建室内房间"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call scene.init() first.")
        ex, ey, roof_h = self.room.extent
        roof_thickness = 0.2
        wall_thickness = 0.2
        # 地板
        self.scene.add_entity(
            morph=ezsim.morphs.Plane(fixed=True, contype=0x0001, conaffinity=0x0001),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['floor'])) 
                if isinstance(self.room.textures['floor'], str) else ezsim.surfaces.Rough(color=self.room.textures['floor']) 
        )      
        # 屋顶
        self.scene.add_entity(
            morph=ezsim.morphs.Mesh(file="meshes/drone_racing/box.obj", pos=(0, 0, roof_h + roof_thickness/8), scale=(ex + 2*wall_thickness, ey + 2*wall_thickness, roof_thickness), fixed=True, contype=0x0002, conaffinity=0x0002),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['roof'])) 
                if isinstance(self.room.textures['roof'], str) else ezsim.surfaces.Rough(color=self.room.textures['roof']) 
        )
        # 墙壁
        # 左墙
        self.scene.add_entity(
            morph=ezsim.morphs.Plane(pos=(0, ey / 2, roof_h / 2), normal=(0, -1, 0), fixed=True, contype=0x0004, conaffinity=0x0004),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['left'])) 
                if isinstance(self.room.textures['left'], str) else ezsim.surfaces.Rough(color=self.room.textures['left'])
        )
        # 右墙
        self.scene.add_entity(
            morph=ezsim.morphs.Plane(pos=(0, -ey / 2, roof_h / 2), normal=(0, 1, 0), fixed=True, contype=0x0004, conaffinity=0x0004),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['right'])) 
                if isinstance(self.room.textures['right'], str) else ezsim.surfaces.Rough(color=self.room.textures['right']) 
        )
        # 前墙
        self.scene.add_entity(
            morph=ezsim.morphs.Plane(pos=(ex / 2, 0, roof_h / 2), normal=(-1, 0, 0), fixed=True, contype=0x0008, conaffinity=0x0008),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['front'])) 
                if isinstance(self.room.textures['front'], str) else ezsim.surfaces.Rough(color=self.room.textures['front']) 
        )
        # 后墙
        self.scene.add_entity(
            morph=ezsim.morphs.Plane(pos=(-ex / 2, 0, roof_h / 2), normal=(1, 0, 0), fixed=True, contype=0x0008, conaffinity=0x0008),
            surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=self.room.textures['back'])) 
                if isinstance(self.room.textures['back'], str) else ezsim.surfaces.Rough(color=self.room.textures['back']) 
        )
        SCENE_STATUS = 1
        return self
    
    # FINISH: 已完成/基本无需检查 室内天花板灯阵和其他辅助光防止出现不真实光影pattern的构建方法
    def _build_lights(self):
        """根据配置构建室内照明系统，包括天花板灯矩阵、墙面补充照明和环境光"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call scene.init() first.")
        ex, ey, roof_h = self.room.extent
        mex, mey = self.lights.ceil['margin']
        avax, avay = ex-2*mex, ey-2*mey
        if avax <= 0 or avay <= 0:
            raise ezsim.logger.warning(f"Warning: Room too small for lighting with margins. Room: {ex}x{ey}, Required: {2*mex}x{2*mey}")
        # 计算天花板灯之间的间距和起始位置
        grid_x, grid_y = self.lights.ceil['grid']
        ceil_lpos_x = [-avax/2 + (i* avax/(grid_x - 1)) for i in range(grid_x)] if grid_x > 1 else [0]
        ceil_lpos_y = [-avay/2 + (j* avay/(grid_y - 1)) for j in range(grid_y)] if grid_y > 1 else [0]

        for clpx in ceil_lpos_x:
            for clpy in ceil_lpos_y:
                pos = (clpx, clpy, self.lights.ceil['height'])
                self.scene.add_light(
                    pos=pos,
                    dir=[0.0, 0.0, -1.0],
                    directional=self.lights.ceil['directional'],
                    castshadow=1,
                    cutoff=self.lights.ceil['cutoff_deg'],
                    intensity=self.lights.ceil['intensity'],
                    attenuation=self.lights.ceil['attenuation']
                )
        # 墙面补充照明 - 在天花板灯的间隔处添加补充光源以消除光影图案
        mwx,mwy = self.lights.wall['margin']
        wlh = self.lights.wall['height']
        wall_lpos_x = [(ceil_lpos_x[i + 1] + ceil_lpos_x[i]) / 2 for i in range(len(ceil_lpos_x) - 1)]
        for wlpx in wall_lpos_x:
            # 左墙补充照明
            self.scene.add_light(
                pos=[wlpx, ey/2 - mwy , wlh],
                dir=[0.0,-1.0,0.0],
                directional=self.lights.wall['directional'],
                castshadow=0,
                cutoff=self.lights.wall['cutoff_deg'],
                intensity=self.lights.wall['intensity'],
                attenuation=self.lights.wall['attenuation']
            )
            # 右墙补充照明
            self.scene.add_light(
                pos=[wlpx, -ey/2 + mwy , wlh],
                dir=[0.0, 1.0, 0.0],
                directional=self.lights.wall['directional'],
                castshadow=0,
                cutoff=self.lights.wall['cutoff_deg'],
                intensity=self.lights.wall['intensity'],
                attenuation=self.lights.wall['attenuation']
            )
        wall_lpos_y = [(ceil_lpos_y[i + 1] + ceil_lpos_y[i]) / 2 for i in range(len(ceil_lpos_y) - 1)]
        for wlpy in wall_lpos_y:
            # 前墙补充照明
            self.scene.add_light(
                pos=[ex/2 - mwx, wlpy, wlh],
                dir=[-1.0, 0.0, 0.0],
                directional=self.lights.wall['directional'],
                castshadow=0,
                cutoff=self.lights.wall['cutoff_deg'],
                intensity=self.lights.wall['intensity'],
                attenuation=self.lights.wall['attenuation']
            )
            # 后墙补充照明
            self.scene.add_light(
                pos=[-ex/2 + mwx, wlpy, wlh],
                dir=[1.0, 0.0, 0.0],
                directional=self.lights.wall['directional'],
                castshadow=0,
                cutoff=self.lights.wall['cutoff_deg'],
                intensity=self.lights.wall['intensity'],
                attenuation=self.lights.wall['attenuation']
            )
        # 添加环境光以进一步平滑光照
        amlight_num = self.lights.ambient['num']
        for i in range(amlight_num):
            angle = i * 2 * np.pi / amlight_num
            radius = min(ex, ey) * 0.3
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)
            
            self.scene.add_light(
                pos=[x_pos, y_pos, self.lights.ambient['height']],
                dir=[0.0, 0.0, -1.0],
                directional=self.lights.ambient['directional'],
                castshadow=0,  # 环境光不产生阴影
                cutoff=self.lights.ambient['cutoff_deg'],
                attenuation= self.lights.ambient['attenuation'],
                intensity= self.lights.ambient['intensity'],
            )
        
        # 统计信息
        total_wall_lights = 2 * len(wall_lpos_x) + 2 * len(wall_lpos_y)
        total_ceiling_lights = grid_x * grid_y
        
        ezsim.logger.info(f"Built lighting system:")
        ezsim.logger.info(f"  - {total_ceiling_lights} ceiling lights ({grid_x}x{grid_y} grid) at {self.lights.ceil['height']:.1f}m")
        ezsim.logger.info(f"  - {total_wall_lights} wall lights ({len(wall_lpos_y)} front/back + {len(wall_lpos_x)} left/right each) at {self.lights.wall['height']:.1f}m")
        ezsim.logger.info(f"  - {amlight_num} ambient lights at {self.lights.ambient['height']:.1f}m")
        ezsim.logger.info(f"  - Wall lights positioned between ceiling light intervals to eliminate shadow patterns") 
        SCENE_STATUS = 2
        return self
        
    # FINISH: 已完成/基本无需检查 室内无人机通过门的构建方法
    def _build_gates(self):
        """根据配置构建室内通过门"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call init_scene(args) first.")
        
        # 遍历方形门配置
        sg_fpath = "meshes/drone_racing/square_gate.obj"
        for gate_name, gate_cfg in self.gates['square'].items():
            # 创建 SquareGate 实例并计算相关参数
            gate = SquareGate(
                pos=gate_cfg['pos'],
                size=gate_cfg['size'],
                direction=gate_cfg['direction'],
                path_ratio=gate_cfg.get('path_ratio', 2.0),
                texture=gate_cfg.get('texture', "textures/sjtu_square_gate.png")
            )
            gate.post_euler(self.robot_size)
            
            self.scene.add_entity(
                morph=ezsim.morphs.Mesh(
                    file=sg_fpath, 
                    pos=gate.pos, 
                    euler=gate.euler, 
                    scale=gate.scale, 
                    fixed=True
                ),
                surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=gate.texture)) 
                    if isinstance(gate.texture, str) else ezsim.surfaces.Rough(color=gate.texture)
            )
            self.occupied_spaces.append({
                'name': gate_name, 
                'type': 'gate_square', 
                'pos': gate.pos,
                'aabb': gate.aabb
            })
            ezsim.logger.info(f"Added square gate '{gate_name}' at {gate.pos} with size {gate.size}")
        
        # 遍历圆形门配置
        cg_fpath = "meshes/drone_racing/circle_gate.obj"
        for gate_name, gate_cfg in self.gates['circle'].items():
            # 创建 CircleGate 实例并计算相关参数
            gate = CircleGate(
                pos=gate_cfg['pos'],
                size=gate_cfg['size'],
                direction=gate_cfg['direction'],
                path_ratio=gate_cfg.get('path_ratio', 2.0),
                # texture=gate_cfg.get('texture', "textures/sjtu_circle_gate.png")
                texture=(0.925,0.015,0.0)
            )
            gate.post_euler(self.robot_size)
            
            self.scene.add_entity(
                morph=ezsim.morphs.Mesh(
                    file=cg_fpath, 
                    pos=gate.pos, 
                    euler=gate.euler, 
                    scale=gate.scale, 
                    fixed=True
                ),
                surface=ezsim.surfaces.Rough(diffuse_texture=ezsim.textures.ImageTexture(image_path=gate.texture)) 
                    if isinstance(gate.texture, str) else ezsim.surfaces.Rough(color=gate.texture)
            )
            self.occupied_spaces.append({
                'name': gate_name, 
                'type': 'gate_circle',
                'pos': gate.pos, 
                'aabb': gate.aabb
            })
            ezsim.logger.info(f"Added circle gate '{gate_name}' at {gate.pos} with size {gate.size}")
        
        global SCENE_STATUS
        SCENE_STATUS = 3
        return self
    
    # FINISH: 已完成/基本无需检查 室内无人机通过门的构建方法
    def prealloc_obstacles(self,fwd_scale:float,):
        """
        预计算/预摆放障碍物：
        - 基于房间与已放置的gate AABB，随机生成符合通道约束的障碍物占用方案。
        - 仅在本函数内进行AABB与通道留空(path_ratio*robot_size)检查；不向场景添加实体。
        - 返回可用于build_obstacles的规范化列表，每个元素包含：
          {
            'name': str,
            'type': 'obstacle_box' | 'obstacle_sphere' | 'obstacle_cylinder' | 'obstacle_combo',
            'aabb': np.ndarray(2,3),
            'parts': [ { 'shape': 'box'|'sphere'|'cylinder', 'pos': (x,y,z), 'scale': (sx,sy,sz)?, 'radius': r?, 'height': h? } ]
          }
        """
        # 0. 基本前置与参数
        if self.scene is None:
            raise ValueError("Scene not initialized. Call init_scene(args) first.")

        ex, ey, ez = self.room.extent
        # 目标数量：以面积近似密度，同时限制最大数量
        target_cnt = min(int(ex * ey * self.obstacles.density), self.obstacles.count)
        if target_cnt <= 0:
            return []

        # 通道留空半宽（逐轴）
        clearance = np.array([self.obstacles.path_ratio * v for v in self.robot_size], dtype=float)

        # 已有占用（来自gate等），仅提取包含aabb的项
        occupied_aabbs = []
        for sp in self.occupied_spaces:
            if 'aabb' in sp:
                aabb = np.array(sp['aabb'], dtype=float)
                if aabb.shape == (2, 3):
                    occupied_aabbs.append(aabb)

        rng = np.random.default_rng()

        # 实用函数（仅在本函数内部使用）
        def rand_pos(sz_half: np.ndarray) -> Tuple[float, float, float]:
            """在考虑尺寸与通道壁垒的情况下随机一个中心位置，使AABB不越界。"""
            margins = sz_half + clearance
            # 房间内有效范围
            x = rng.uniform(-ex / 2 + margins[0], ex / 2 - margins[0]) if ex > 2 * margins[0] else 0.0
            y = rng.uniform(-ey / 2 + margins[1], ey / 2 - margins[1]) if ey > 2 * margins[1] else 0.0
            z = rng.uniform(0 + margins[2], ez - margins[2]) if ez > 2 * margins[2] else max(min(ez / 2, ez - margins[2]), margins[2])
            return (float(x), float(y), float(z))

        def expand_aabb(aabb: np.ndarray, pad: np.ndarray) -> np.ndarray:
            return np.stack([aabb[0] - pad, aabb[1] + pad], axis=0)

        def aabb_intersects(a: np.ndarray, b: np.ndarray) -> bool:
            return not (a[1][0] < b[0][0] or a[0][0] > b[1][0] or
                        a[1][1] < b[0][1] or a[0][1] > b[1][1] or
                        a[1][2] < b[0][2] or a[0][2] > b[1][2])

        def valid_with_clearance(test_aabb: np.ndarray) -> Tuple[bool, np.ndarray]:
            """基于带clearance的AABB进行检测并返回(passed, padded_aabb)。
            注意：occupied_spaces中aabb已包含clearance，这里将当前待测AABB也扩张后再判断，
            并以扩张后的AABB作为障碍物的占用空间记录。
            """
            padded = expand_aabb(test_aabb, clearance)
            # 房间内边界检查（直接对padded检查）
            room_min = np.array([-ex / 2, -ey / 2, 0.0])
            room_max = np.array([ex / 2, ey / 2, ez])
            if np.any(padded[0] < room_min) or np.any(padded[1] > room_max):
                return False, padded
            # 与已占用的AABB检查（occupied_aabbs已是带clearance的）
            for occ in occupied_aabbs:
                if aabb_intersects(padded, occ):
                    return False, padded
            return True, padded

        def sample_scales(scale:float=4.0) -> np.ndarray:
            """在[1/scale,scale]中采样各向异性缩放，避免三个维度同时极端。"""
            for _ in range(8):
                s = rng.uniform(1.0/scale, scale, size=(3,))
                if not (np.all(s >= 0.9*scale) or np.all(s <= 1.1/scale)):
                    return s
            # 兜底：给一点多样性
            s = np.array([rng.uniform(0.4, 2.5), rng.uniform(0.4, 2.5), rng.uniform(0.3, 3.0)])
            return s

        def make_single_obstacle(scale:float=4.0) -> Optional[dict]:
            """生成单体障碍物（box/sphere/cylinder），返回规范化描述或None。"""
            # 将未知类型映射到支持类型
            assert scale>1.0, "随机尺度需要大于 1.0"
            type_candidates = [t for t in self.obstacles.types if t in ('box', 'sphere', 'cylinder')]
            if not type_candidates:
                type_candidates = ['box', 'sphere', 'cylinder']
            shape = rng.choice(type_candidates)

            scales = sample_scales(scale)
            if shape == 'box':
                sz_half = 0.5 * scales
                pos = rand_pos(sz_half)
                aabb = np.stack([np.array(pos) - sz_half, np.array(pos) + sz_half], axis=0)
                ok, padded = valid_with_clearance(aabb)
                if not ok:
                    return None
                part = {
                    'shape': 'box',
                    'pos': pos,
                    'scale': tuple(float(v) for v in scales)
                }
                return {
                    'type': 'obstacle_box',
                    'aabb': padded,
                    'parts': [part]
                }
            elif shape == 'sphere':
                # 以最小尺度定义球半径，保持在范围内
                radius = float(max(0.05, 0.5 * float(np.min(scales))))
                sz_half = np.array([radius, radius, radius])
                pos = rand_pos(sz_half)
                aabb = np.stack([np.array(pos) - sz_half, np.array(pos) + sz_half], axis=0)
                ok, padded = valid_with_clearance(aabb)
                if not ok:
                    return None
                # 统一提供scale以便使用预生成带UV的obj
                part = {
                    'shape': 'sphere',
                    'pos': pos,
                    'radius': radius,
                    'scale': (2.0 * radius, 2.0 * radius, 2.0 * radius)
                }
                return {
                    'type': 'obstacle_sphere',
                    'aabb': padded,
                    'parts': [part]
                }
            else:  # cylinder
                # 基于X/Y尺度决定半径，Z尺度为高度
                radius = float(max(0.05, 0.5 * float(np.clip(min(scales[0], scales[1]), 0.1, 10.0))))
                height = float(max(0.1, float(np.clip(scales[2], 0.1, 10.0))))
                sz_half = np.array([radius, radius, 0.5 * height])
                pos = rand_pos(sz_half)
                aabb = np.stack([np.array(pos) - sz_half, np.array(pos) + sz_half], axis=0)
                ok, padded = valid_with_clearance(aabb)
                if not ok:
                    return None
                part = {
                    'shape': 'cylinder',
                    'pos': pos,
                    'radius': radius,
                    'height': height,
                    'scale': (2.0 * radius, 2.0 * radius, height)
                }
                return {
                    'type': 'obstacle_cylinder',
                    'aabb': padded,
                    'parts': [part]
                }

        def make_combo_obstacle() -> Optional[dict]:
            """生成简单组合障碍物（如台灯/小车风格），组内不检查通道，组对外检查通道。"""
            # 50% 生成“台灯”风格（立柱+顶帽），50% 生成“小车”风格（底盘+4轮）
            if rng.random() < 0.5:
                # 台灯：细长柱 + 顶部较扁的“灯罩”（用box或短柱体近似）
                stem_r = float(rng.uniform(0.05, 0.25))
                stem_h = float(rng.uniform(0.5, 2.0))
                shade_w = float(rng.uniform(0.3, 1.0))
                shade_d = float(rng.uniform(0.3, 1.0))
                shade_h = float(rng.uniform(0.1, 0.5))
                # 组AABB半尺寸：灯罩尺寸与立柱综合
                sz_half = np.array([max(stem_r, shade_w / 2), max(stem_r, shade_d / 2), 0.5 * (stem_h + shade_h)])
                pos = rand_pos(sz_half)
                base = np.array(pos)
                parts = [
                    {'shape': 'cylinder', 'pos': tuple(base + np.array([0.0, 0.0, stem_h / 2 - sz_half[2] + 0.5 * (stem_h + shade_h)])), 'radius': stem_r, 'height': stem_h},
                    {'shape': 'box', 'pos': tuple(base + np.array([0.0, 0.0, (stem_h + shade_h) / 2 - sz_half[2]])), 'scale': (shade_w, shade_d, shade_h)}
                ]
                # 计算组AABB
                mins = []
                maxs = []
                for p in parts:
                    if p['shape'] == 'cylinder':
                        r = p['radius']; h = p['height']
                        c = np.array([r, r, h / 2])
                    elif p['shape'] == 'box':
                        s = np.array(p['scale']) / 2
                        c = s
                    else:  # sphere（不在当前组合里，但留兜底）
                        r = p.get('radius', 0.2)
                        c = np.array([r, r, r])
                    pp = np.array(p['pos'])
                    mins.append(pp - c)
                    maxs.append(pp + c)
                aabb = np.stack([np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)], axis=0)
                ok, padded = valid_with_clearance(aabb)
                if not ok:
                    return None
                # 为cylinder部件附带scale，便于使用UV obj
                for p in parts:
                    if p['shape'] == 'cylinder' and 'scale' not in p:
                        p['scale'] = (2.0 * p['radius'], 2.0 * p['radius'], p['height'])
                return {
                    'type': 'obstacle_combo_lamp',
                    'aabb': padded,
                    'parts': parts
                }
            else:
                # 小车：底盘box + 4个竖直轮子cylinder（简化）
                body_l = float(rng.uniform(0.6, 2.0))
                body_w = float(rng.uniform(0.4, 1.2))
                body_h = float(rng.uniform(0.2, 0.6))
                wheel_r = float(rng.uniform(0.1, 0.3))
                wheel_h = float(rng.uniform(0.1, 0.25))
                sz_half = np.array([0.5 * body_l, 0.5 * body_w, 0.5 * (body_h + wheel_h)])
                pos = rand_pos(sz_half)
                base = np.array(pos)
                # 轮子位置（竖直，贴近底盘四角）
                offsets = [
                    np.array([ body_l/2 - wheel_r,  body_w/2 - wheel_r, - (body_h/2) + wheel_h/2]),
                    np.array([-body_l/2 + wheel_r,  body_w/2 - wheel_r, - (body_h/2) + wheel_h/2]),
                    np.array([ body_l/2 - wheel_r, -body_w/2 + wheel_r, - (body_h/2) + wheel_h/2]),
                    np.array([-body_l/2 + wheel_r, -body_w/2 + wheel_r, - (body_h/2) + wheel_h/2]),
                ]
                parts = [
                    {'shape': 'box', 'pos': tuple(base), 'scale': (body_l, body_w, body_h)},
                ] + [
                    {'shape': 'cylinder', 'pos': tuple(base + off), 'radius': wheel_r, 'height': wheel_h} for off in offsets
                ]
                mins = []
                maxs = []
                for p in parts:
                    if p['shape'] == 'cylinder':
                        r = p['radius']; h = p['height']
                        c = np.array([r, r, h / 2])
                    elif p['shape'] == 'box':
                        s = np.array(p['scale']) / 2
                        c = s
                    else:
                        r = p.get('radius', 0.2)
                        c = np.array([r, r, r])
                    pp = np.array(p['pos'])
                    mins.append(pp - c)
                    maxs.append(pp + c)
                aabb = np.stack([np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)], axis=0)
                ok, padded = valid_with_clearance(aabb)
                if not ok:
                    return None
                for p in parts:
                    if p['shape'] == 'cylinder' and 'scale' not in p:
                        p['scale'] = (2.0 * p['radius'], 2.0 * p['radius'], p['height'])
                return {
                    'type': 'obstacle_combo_car',
                    'aabb': padded,
                    'parts': parts
                }

        # 逐次尝试填充
        max_attempts = max(200, target_cnt * 25)
        result: List[dict] = []
        attempts = 0
        while len(result) < target_cnt and attempts < max_attempts:
            attempts += 1
            # 少量概率生成组合体，提升场景多样性
            maker = make_combo_obstacle if rng.random() < -1 else make_single_obstacle
            ob = maker()
            if ob is None:
                continue
            # 通过则纳入结果，并纳入占用集合（用于后续约束）
            result.append({
                'name': f"obs_{len(result)}",
                **ob
            })
            occupied_aabbs.append(ob['aabb'])

        # 缓存并返回
        self._prealloc_obstacles = result
        ezsim.logger.warning(f"Preallocated {len(result)}/{target_cnt} obstacles with clearance.")
        return result
    
    # TODO:, not ready
    def build_obstacles(self):
        """根据预摆放结果将障碍物实体添加到场景，并登记occupied_spaces。"""
        if self.scene is None:
            raise ValueError("Scene not initialized. Call init_scene(args) first.")

        # 获取或生成预摆放数据
        obstacles = getattr(self, '_prealloc_obstacles', None)
        if obstacles is None:
            obstacles = self.prealloc_obstacles()

        # 颜色生成器（柔和随机色）
        rng = np.random.default_rng()
        def rand_color():
            base = rng.uniform(0.2, 0.9, size=(3,))
            return (float(base[0]), float(base[1]), float(base[2]), 1.0)

        # 将每个障碍物的所有part添加到场景
        for ob in obstacles:
            parts = ob.get('parts', [])
            for p in parts:
                shape = p['shape']
                color = rand_color()
                if shape == 'box':
                    # 使用已存在的通用box.obj，并通过scale控制尺寸
                    self.scene.add_entity(
                        morph=ezsim.morphs.Mesh(file="meshes/drone_racing/box.obj", pos=p['pos'], scale=p['scale'], fixed=True),
                        surface=ezsim.surfaces.Rough(color=color)
                    )
                elif shape == 'sphere':
                    scale = p.get('scale')
                    if scale is None:
                        r = p.get('radius', 0.2)
                        scale = (2.0 * r, 2.0 * r, 2.0 * r)
                    self.scene.add_entity(
                        morph=ezsim.morphs.Mesh(file="meshes/drone_racing/sphere.obj", pos=p['pos'], scale=scale, fixed=True),
                        surface=ezsim.surfaces.Rough(color=color)
                    )
                elif shape == 'cylinder':
                    # 若存在Cylinder原语则优先，否则退化为box近似
                    scale = p.get('scale')
                    if scale is None:
                        r = p.get('radius', 0.2)
                        h = p.get('height', 0.5)
                        scale = (2.0 * r, 2.0 * r, h)
                    try:
                        self.scene.add_entity(
                            morph=ezsim.morphs.Mesh(file="meshes/drone_racing/cylinder.obj", pos=p['pos'], scale=scale, fixed=True),
                            surface=ezsim.surfaces.Rough(color=color)
                        )
                    except Exception:
                        # 退化为box近似（直径x直径x高度）
                        self.scene.add_entity(
                            morph=ezsim.morphs.Mesh(file="meshes/drone_racing/box.obj", pos=p['pos'], scale=scale, fixed=True),
                            surface=ezsim.surfaces.Rough(color=color)
                        )

            # 记录占用空间（用于后续生成/调试）
            self.occupied_spaces.append({
                'name': ob.get('name', 'obstacle'),
                'type': ob.get('type', 'obstacle'),
                'pos': tuple((ob['aabb'][0] + ob['aabb'][1]) / 2.0),
                'aabb': ob['aabb']
            })

        ezsim.logger.info(f"Built {len(obstacles)} obstacles into the scene.")


##################################
# main
##################################

def main():
    ##########################################
    # args
    ##########################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default='examples/worldgen/indoor_config.yaml')
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-b", "--n_envs", type=int, default=3)
    parser.add_argument("-s", "--n_steps", type=int, default=2)
    parser.add_argument("-r", "--render_all_cameras", action="store_true", default=False)
    parser.add_argument("-o", "--output_dir", type=str, default="img_output/test")
    parser.add_argument("-u", "--use_rasterizer", action="store_true", default=False)
    parser.add_argument("-d", "--debug_occupied_spaces", action="store_true", default=False, help="Draw debug wireframe boxes for occupied spaces")

    # video recording options
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--w", type=int, default=1280, help="Camera width")
    parser.add_argument("--h", type=int, default=960, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_len", type=int, default=5, help="Video length in seconds")
    args = parser.parse_args()

    ##########################################
    # backend init
    ##########################################
    SCENE_STATUS, SCENE = init_scene(args)

    ##########################################
    # indoor scene init/construct/build/run
    ##########################################
    indoor_dojo = IndoorScene(yaml_path=args.file)  
    indoor_dojo.construct(debug_occupied_spaces=args.debug_occupied_spaces)  # 根据命令行参数决定是否启用调试
    indoor_dojo.build(n_envs=args.n_envs)
    indoor_dojo.run(steps=args.n_steps, w_rgb=True, w_depth=True, w_normal=True)


if __name__ == "__main__":
    main()
