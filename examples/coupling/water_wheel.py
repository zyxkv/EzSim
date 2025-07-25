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
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.5),
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
            max_FPS=120,
        ),
        vis_options=ezsim.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=args.vis,
        sph_options=ezsim.options.SPHOptions(
            particle_size=0.02,
        ),
    )

    plane = scene.add_entity(ezsim.morphs.Plane())
    wheel_0 = scene.add_entity(
        morph=ezsim.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )

    emitter = scene.add_emitter(
        material=ezsim.materials.SPH.Liquid(sampler="regular"),
        max_particles=100000,
        surface=ezsim.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=5)

    horizon = 500
    for i in range(horizon):
        print(i)
        emitter.emit(
            pos=np.array([0.5, 1.0, 3.5]),
            direction=np.array([0.0, 0, -1.0]),
            speed=5.0,
            droplet_shape="circle",
            droplet_size=0.22,
        )
        scene.step()


if __name__ == "__main__":
    main()
