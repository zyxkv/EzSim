import numpy as np
import ezsim
import argparse

from ezsim.sensors import SensorDataRecorder, VideoFileWriter


def build_outdoor_dojo():
    pass  # Placeholder for outdoor dojo setup

def build_indoor_dojo():
    pass 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--w", type=int, default=640, help="Camera width")
    parser.add_argument("--h", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_len", type=int, default=5, help="Video length in seconds")
    args = parser.parse_args()

    ezsim.init(
        backend=ezsim.gpu,
        precision="32",
        seed=4320,
        eps=1e-12,
        log_time=False
    )

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(dt=args.dt),
        rigid_options=ezsim.options.RigidOptions(
            box_box_detection=False,
            max_collision_pairs=1000,
            use_gjk_collision=True,
            enable_mujoco_compatibility=False,
        ),
        vis_options=ezsim.options.VisOptions(show_world_frame=True),
        show_viewer=False,
    )

    # Add entities and sensors here...