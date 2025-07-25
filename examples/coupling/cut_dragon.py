import argparse

import numpy as np

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=ezsim.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.01),
            upper_bound=(1.0, 1.0, 2.0),
            grid_density=64,
            enable_CPIC=True,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(1.2, 0.9, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=35,
            max_FPS=120,
        ),
        show_viewer=args.vis,
        vis_options=ezsim.options.VisOptions(
            visualize_mpm_boundary=True,
            # rendered_envs_idx=[2],
        ),
    )

    plane = scene.add_entity(
        material=ezsim.materials.Rigid(),
        morph=ezsim.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    cutter = scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/cross_cutter.obj",
            euler=(90, 0, 0),
            scale=0.8,
            pos=(0.0, 0.0, 0.3),
            fixed=True,
            convexify=False,
        ),
        surface=ezsim.surfaces.Iron(),
    )
    dragon = scene.add_entity(
        material=ezsim.materials.MPM.Elastic(sampler="pbs-64"),
        morph=ezsim.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.007,
            euler=(0, 0, 90),
            pos=(0.3, -0.0, 1.3),
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.6, 1.0, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    scene.build(n_envs=5)

    horizon = 400
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
