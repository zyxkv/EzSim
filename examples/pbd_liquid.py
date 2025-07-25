import argparse

import numpy as np

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=0, precision="32", logging_level="debug")

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=2e-3,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        pbd_options=ezsim.options.PBDOptions(
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
        ),
    )

    ########################## entities ##########################

    liquid = scene.add_entity(
        material=ezsim.materials.PBD.Liquid(rho=1.0, density_relaxation=1.0, viscosity_relaxation=0.0, sampler="regular"),
        morph=ezsim.morphs.Box(lower=(0.2, 0.1, 0.1), upper=(0.4, 0.3, 0.5)),
    )
    scene.build(n_envs=5)

    for i in range(10000):
        scene.step()


if __name__ == "__main__":
    main()
