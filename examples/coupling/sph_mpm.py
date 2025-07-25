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
            dt=2e-3,
            substeps=10,
        ),
        vis_options=ezsim.options.VisOptions(
            visualize_sph_boundary=True,
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        ),
        mpm_options=ezsim.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.1),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        sph_options=ezsim.options.SPHOptions(
            lower_bound=(-0.03, -0.03, -0.08),
            upper_bound=(1.03, 1.03, 1.0),
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(0.8, -3, 1.42),
            camera_lookat=(0.5, 0.5, 0.4),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        morph=ezsim.morphs.Plane(),
    )
    water = scene.add_entity(
        material=ezsim.materials.SPH.Liquid(),
        morph=ezsim.morphs.Box(
            pos=(0.4, 0.5, 0.25),
            size=(0.7, 0.9, 0.5),
        ),
        surface=ezsim.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="particle",
        ),
    )

    duck = scene.add_entity(
        material=ezsim.materials.MPM.Elastic(rho=200),
        morph=ezsim.morphs.Mesh(
            file="meshes/duck.obj",
            pos=(0.5, 0.5, 0.7),
            scale=0.07,
            euler=(90.0, 0.0, 90.0),
        ),
        surface=ezsim.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
            vis_mode="particle",
        ),
    )
    ########################## build ##########################
    scene.build(n_envs=5)

    for i in range(800):
        scene.step()


if __name__ == "__main__":
    main()
