import numpy as np
import ezsim
import argparse
import time 
from tqdm import tqdm

from ezsim.sensors import SensorDataRecorder, VideoFileWriter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygltflib")

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
parser.add_argument("--w", type=int, default=640, help="Camera width")
parser.add_argument("--h", type=int, default=480, help="Camera height")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--video_len", type=int, default=5, help="Video length in seconds")
args = parser.parse_args()

test_glb = 'ext_assets/stone_gate.glb'

ezsim.init(
    backend=ezsim.gpu, 
    precision="32", 
    seed=4320, 
    eps=1e-12,
    log_time=False
)

scene = ezsim.Scene(
    sim_options=ezsim.options.SimOptions(
        dt=args.dt,
    ),
    rigid_options=ezsim.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
    ),
    vis_options=ezsim.options.VisOptions(
        show_world_frame=True,
    ),
    show_viewer=False,
)

########################## add entity ##########################
plane = scene.add_entity(ezsim.morphs.Plane(pos=(0, 0, 0)))
glb_entity = scene.add_entity(
    ezsim.morphs.Mesh(
        file=test_glb,
        fixed=True,  # Make it static
        euler=(-90, 180, 0),  # only if needed
        pos=(0, 1, 0.0),  # only if needed
        scale=(0.2, 0.2, 0.2),  # Scale down for better visibility
        # 添加凸包选项
        
        # convex_hull_threshold=10.0,  # 增加容忍度
        # use_convex_decomposition=False,  # 强制使用简单凸包
    )
)
########################## add sensors ##########################
data_recorder = SensorDataRecorder(step_dt=args.dt)
# Add camera for visualization
cam = scene.add_camera(
    res=(args.w, args.h),
    pos=(5, 0, 3),
    lookat=(0.0, 0.0, 0.5),
    fov=45,
    GUI=False,
)

# we can also record the camera video using data_recorder
data_recorder.add_sensor(cam, VideoFileWriter(filename="stone_gate_glb.mp4"))
########################## build ##########################
scene.build()
data_recorder.start_recording()
try:
    for i in range(args.video_len*args.fps):
        scene.step()
        cam.set_pose(
            pos    = (5 * np.sin(i / args.fps), 5 * np.cos(i / args.fps), 3),
            lookat = (0, 0, 0.5),
        )
        data_recorder.step()
        # time.sleep(0.01)


except KeyboardInterrupt:
    ezsim.logger.info("Simulation interrupted, exiting.")
finally:
    ezsim.logger.info("Simulation finished.")

    data_recorder.stop_recording()