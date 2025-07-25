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
            dt=3e-3,
            substeps=10,
        ),
        mpm_options=ezsim.options.MPMOptions(
            lower_bound=(0.0, -1.0, -0.1),
            upper_bound=(0.57, 1.0, 2.4),
            grid_density=64,
        ),
        show_viewer=args.vis,
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(4.5, 0.0, 1.42),
            camera_lookat=(1.0, 0.0, 1.0),
            camera_fov=30,
            max_FPS=120,
        ),
        vis_options=ezsim.options.VisOptions(
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        ),
    )

    plane = scene.add_entity(
        material=ezsim.materials.Rigid(
            needs_coup=True,
            coup_friction=0.2,
        ),
        morph=ezsim.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    mat_wheel = ezsim.materials.Rigid(
        needs_coup=True,
        coup_softness=0.0,
    )
    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=ezsim.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, -0.2, 1.6),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=ezsim.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, 0.3, 1.2),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=ezsim.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, -0.3, 0.8),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    wheel_0 = scene.add_entity(
        material=mat_wheel,
        morph=ezsim.morphs.URDF(
            file="urdf/wheel/wheel.urdf",
            pos=(0.5, 0.4, 0.4),
            euler=(0, 0, 90),
            scale=0.6,
            convexify=False,
            fixed=True,
        ),
    )

    emitter = scene.add_emitter(
        material=ezsim.materials.MPM.Sand(),
        max_particles=200000,
        surface=ezsim.surfaces.Rough(
            color=(1.0, 0.9, 0.6, 1.0),
        ),
    )
    scene.build(n_envs=5)

    horizon = 1000
    for i in range(horizon):
        print(i)
        emitter.emit(
            pos=np.array([0.5, 0.0, 2.3]),
            direction=np.array([0.0, np.sin(i / 10) * 0.35, -1.0]),
            speed=8.0,
            droplet_shape="rectangle",
            droplet_size=[0.03, 0.05],
        )
        scene.step()


if __name__ == "__main__":
    main()
