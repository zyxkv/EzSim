import argparse

import numpy as np

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(backend=ezsim.cpu if args.cpu else ezsim.gpu, logging_level="debug")

    ########################## create a scene ##########################

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(2, 2, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_up=(0, 0, 1),
        ),
        show_viewer=args.vis,
    )

    ########################## materials ##########################
    mat_elastic = ezsim.materials.PBD.Elastic()

    ########################## entities ##########################

    bunny = scene.add_entity(
        material=mat_elastic,
        morph=ezsim.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.003,
            pos=(0, 0, 0.8),
        ),
        surface=ezsim.surfaces.Default(
            # vis_mode='recon',
        ),
    )
    ########################## build ##########################
    scene.build()

    horizon = 1000
    # forward pass
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
