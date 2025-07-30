import numpy as np
import ezsim
import argparse
import time 
from tqdm import tqdm

from ezsim.sensors import SensorDataRecorder, VideoFileWriter

pile_type = "static"
num_cubes = 5

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
parser.add_argument("--pile_type", type=str, default=pile_type, choices=["static", "falling"])
parser.add_argument("--num_cubes", type=int, default=num_cubes, choices=range(5, 11))
parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU")
args = parser.parse_args()

pile_type = args.pile_type
num_cubes = args.num_cubes
cpu = args.cpu
backend = ezsim.cpu if cpu else ezsim.gpu

ezsim.init(backend=backend, precision="32")

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
    viewer_options=ezsim.options.ViewerOptions(
        camera_pos=(0, -5.5, 2.5),
        camera_lookat=(0, 0.0, 1.5),
        camera_fov=30,
        max_FPS=60,
    ),
    vis_options=ezsim.options.VisOptions(
        show_world_frame=False,
    ),
    show_viewer=False,
)

plane = scene.add_entity(ezsim.morphs.Plane(pos=(0, 0, 0)))



# create pyramid of boxes
box_size = 0.25
if pile_type == "static":
    box_spacing = box_size
else:
    box_spacing = 1.1 * box_size
vec_one = np.array([1.0, 1.0, 1.0])
box_pos_offset = (0 - 0.5, 1, 0.0) + 0.5 * box_size * vec_one
boxes = {}
for i in range(num_cubes):
    for j in range(num_cubes - i):
        box = scene.add_entity(
            ezsim.morphs.Box(
                size=box_size * vec_one, 
                pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j])),
        )

#create a sphere to collide with the pyramid
sphere_radius = 1.0 * box_size
sphere = scene.add_entity(
    morph=ezsim.morphs.Sphere(
        radius=sphere_radius, 
        pos=(0.0, 0.0, num_stacks * (height_offset + box_height) + 5 * sphere_radius)
    ),
)
########################## add sensors ##########################
data_recorder = SensorDataRecorder(step_dt=args.dt)
# Add camera for visualization
cam = scene.add_camera(
    res=(640, 480),
    pos=(0, -5.5, 2.5),
    lookat=(0.0, 0.0, 1.5),
    fov=45,
    GUI=False,
)

# we can also record the camera video using data_recorder
data_recorder.add_sensor(cam, VideoFileWriter(filename="pyramid.mp4"))
########################## build ##########################
scene.build()
data_recorder.start_recording()
try:
    for i in range(500):
        scene.step()
        data_recorder.step()
        # time.sleep(0.01)


except KeyboardInterrupt:
    ezsim.logger.info("Simulation interrupted, exiting.")
finally:
    ezsim.logger.info("Simulation finished.")

    data_recorder.stop_recording()