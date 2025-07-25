import argparse

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
            dt=1e-2,
            substeps=10,
        ),
        sph_options=ezsim.options.SPHOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 2.4),
        ),
        vis_options=ezsim.options.VisOptions(
            visualize_sph_boundary=True,
            rendered_envs_idx=[0],
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3.5, -3.15, 2.42),
            camera_lookat=(0.5, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = ezsim.materials.Rigid(needs_coup=True, coup_friction=0.0)
    plane = scene.add_entity(
        morph=ezsim.morphs.Plane(),
    )
    water = scene.add_entity(
        material=ezsim.materials.SPH.Liquid(mu=0.01, sampler="regular"),
        morph=ezsim.morphs.Box(
            pos=(0.5, 0.0, 0.6),
            size=(0.9, 1.6, 1.2),
        ),
        surface=ezsim.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=ezsim.morphs.Box(
            pos=(0.5, 0.0, 2.4),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=False,
        ),
    )

    ########################## build ##########################
    scene.build()

    for i in range(5000):
        scene.step()


if __name__ == "__main__":
    main()
