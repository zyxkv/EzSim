import argparse
import time

import numpy as np
import torch

import ezsim
from ezsim.sensors import SensorDataRecorder, VideoFileWriter



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--w", type=int, default=640, help="Camera width")
    parser.add_argument("--h", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_len", type=int, default=5, help="Video length in seconds")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=4320, backend=ezsim.cpu if args.cpu else ezsim.gpu)

    ########################## create a scene ##########################

    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=args.dt,
            constraint_solver=ezsim.constraint_solver.Newton,
        ),
        # viewer_options=ezsim.options.ViewerOptions(
        #     camera_pos=(-5.0, -5.0, 10.0),
        #     camera_lookat=(5.0, 5.0, 0.0),
        #     camera_fov=40,
        # ),
        show_viewer=args.vis,
    )

    horizontal_scale = 0.25
    vertical_scale = 0.005
    ########################## entities ##########################
    terrain = scene.add_entity(
        morph=ezsim.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            subterrain_types=[
                ["flat_terrain", "random_uniform_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
            ],
        ),
    )
    ball = scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(1.0, 1.0, 1.0),
            radius=0.1,
        ),
    )
    ########################## add sensors ##########################
    data_recorder = SensorDataRecorder(step_dt=args.dt)
    # Add camera for visualization
    cam = scene.add_camera(
        res=(args.w, args.h),
        pos=(-5.0, -5.0, 10.0),
        lookat=(5.0, 5.0, 0.0),
        fov=40,
        GUI=False,
    )

    # we can also record the camera video using data_recorder
    data_recorder.add_sensor(cam, VideoFileWriter(filename="random_terrain_sub.mp4"))

    ########################## build ##########################
    scene.build(n_envs=100)

    ball.set_pos(torch.cartesian_prod(*(torch.arange(1, 11),) * 2, torch.tensor((1,))))

    (terrain_geom,) = terrain.geoms
    height_field = terrain_geom.metadata["height_field"]
    rows = (horizontal_scale * torch.arange(height_field.shape[0])).reshape((-1, 1)).expand(height_field.shape)
    cols = (horizontal_scale * torch.arange(height_field.shape[1])).reshape((1, -1)).expand(height_field.shape)
    heights = vertical_scale * torch.as_tensor(height_field)
    poss = torch.stack((rows, cols, heights), dim=-1).reshape((-1, 3))
    scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0.0, 0.0, 1.0, 0.7))
    data_recorder.start_recording()
    try:
        for _ in range(args.video_len*args.fps):
            scene.step()
            data_recorder.step()
            time.sleep(1/args.fps)
    except KeyboardInterrupt:
        ezsim.logger.info("Simulation interrupted, exiting.")
    finally:
        ezsim.logger.info("Simulation finished.")

        data_recorder.stop_recording()


if __name__ == "__main__":
    main()
