import argparse
import time

import numpy as np
import torch

import ezsim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=0, backend=ezsim.cpu if args.cpu else ezsim.gpu)

    ########################## create a scene ##########################

    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            constraint_solver=ezsim.constraint_solver.Newton,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
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
    ########################## build ##########################
    scene.build(n_envs=100)

    ball.set_pos(torch.cartesian_prod(*(torch.arange(1, 11),) * 2, torch.tensor((1,))))

    height_field = terrain.geoms[0].metadata["height_field"]
    rows = horizontal_scale * torch.range(0, height_field.shape[0] - 1, 1).unsqueeze(1).repeat(
        1, height_field.shape[1]
    ).unsqueeze(-1)
    cols = horizontal_scale * torch.range(0, height_field.shape[1] - 1, 1).unsqueeze(0).repeat(
        height_field.shape[0], 1
    ).unsqueeze(-1)
    heights = vertical_scale * torch.tensor(height_field).unsqueeze(-1)

    poss = torch.cat([rows, cols, heights], dim=-1).reshape(-1, 3)
    scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0, 0, 1, 0.7))
    for _ in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
