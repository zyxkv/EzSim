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
            substeps=20,
        ),
        mpm_options=ezsim.options.MPMOptions(
            lower_bound=(-0.45, -0.65, -0.01),
            upper_bound=(0.45, 0.65, 1.0),
            grid_density=64,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(4.5, 1.0, 1.42),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=22,
            max_FPS=120,
        ),
        show_viewer=args.vis,
        vis_options=ezsim.options.VisOptions(
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        ),
    )

    plane = scene.add_entity(morph=ezsim.morphs.Plane())
    cube0 = scene.add_entity(
        material=ezsim.materials.MPM.Elastic(rho=400),
        morph=ezsim.morphs.Box(
            pos=(0.0, 0.25, 0.4),
            size=(0.12, 0.12, 0.12),
        ),
        surface=ezsim.surfaces.Rough(
            color=(1.0, 0.5, 0.5, 1.0),
            vis_mode="particle",
        ),
    )

    cube0 = scene.add_entity(
        material=ezsim.materials.MPM.Elastic(rho=400),
        morph=ezsim.morphs.Sphere(
            pos=(0.15, 0.45, 0.5),
            radius=0.06,
        ),
        surface=ezsim.surfaces.Rough(
            color=(1.0, 1.0, 0.5, 1.0),
            vis_mode="particle",
        ),
    )

    cube0 = scene.add_entity(
        material=ezsim.materials.MPM.Elastic(rho=400),
        morph=ezsim.morphs.Cylinder(
            pos=(-0.15, 0.45, 0.6),
            radius=0.05,
            height=0.14,
        ),
        surface=ezsim.surfaces.Rough(
            color=(0.5, 1.0, 1.0, 1.0),
            vis_mode="particle",
        ),
    )
    emitter1 = scene.add_emitter(
        material=ezsim.materials.MPM.Liquid(sampler="random"),
        max_particles=800000,
        surface=ezsim.surfaces.Rough(
            color=(0.0, 0.9, 0.4, 1.0),
        ),
    )
    emitter2 = scene.add_emitter(
        material=ezsim.materials.MPM.Liquid(sampler="random"),
        max_particles=800000,
        surface=ezsim.surfaces.Rough(
            color=(0.0, 0.4, 0.9, 1.0),
        ),
    )
    scene.build(n_envs=5)

    horizon = 1000
    for i in range(horizon):
        if i < 500:
            emitter1.emit(
                pos=np.array([0.16, -0.4, 0.5]),
                direction=np.array([0.0, 0.0, -1.0]),
                speed=1.5,
                droplet_shape="circle",
                droplet_size=0.16,
            )
            emitter2.emit(
                pos=np.array([-0.16, -0.4, 0.5]),
                direction=np.array([0.0, 0.0, -1.0]),
                speed=1.5,
                droplet_shape="circle",
                droplet_size=0.16,
            )
        scene.step()


if __name__ == "__main__":
    main()
