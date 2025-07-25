import argparse

import numpy as np

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=0, precision="32", logging_level="debug", backend=ezsim.cpu if args.cpu else ezsim.gpu)

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=ezsim.options.PBDOptions(
            particle_size=1e-2,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=ezsim.options.VisOptions(
            rendered_envs_idx=[1],
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = ezsim.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=ezsim.morphs.Plane(),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=ezsim.morphs.Box(
            pos=(0.5, 0.5, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=True,
        ),
    )

    cloth = scene.add_entity(
        material=ezsim.materials.PBD.Cloth(),
        morph=ezsim.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=1.0,
            pos=(0.5, 0.5, 0.5),
            euler=(180.0, 0.0, 0.0),
        ),
        surface=ezsim.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=5)

    horizon = 500

    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
