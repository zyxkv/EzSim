import numpy as np
import pytest

import ezsim

from .utils import assert_allclose


def test_rigid_mpm_muscle(show_viewer):
    ball_pos_init = (0.8, 0.6, 0.12)

    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        rigid_options=ezsim.options.RigidOptions(
            gravity=(0, 0, -9.8),
            constraint_timeconst=0.02,
        ),
        mpm_options=ezsim.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
            gravity=(0.0, 0.0, 0.0),  # mimic gravity compensation
            enable_CPIC=True,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(1.5, 1.3, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=ezsim.morphs.Plane())
    robot = scene.add_entity(
        morph=ezsim.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.5, 0.5, 0.3),
            euler=(0.0, 0.0, 0.0),
            scale=0.2,
            fixed=True,
        ),
        material=ezsim.materials.Hybrid(
            mat_rigid=ezsim.materials.Rigid(
                gravity_compensation=1.0,
            ),
            mat_soft=ezsim.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=1000.0,
                model="neohooken",
            ),
            thickness=0.05,
            damping=1000.0,
        ),
    )
    ball = scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=ball_pos_init,
            radius=0.12,
        ),
        material=ezsim.materials.Rigid(rho=1000, friction=0.5),
    )
    scene.build()

    scene.reset()
    for i in range(370):
        robot.control_dofs_velocity(np.array([1.0 * np.sin(2 * np.pi * i * 0.001)] * robot.n_dofs))
        scene.step()

    with np.testing.assert_raises(AssertionError):
        assert_allclose(ball.get_pos(), ball_pos_init, atol=1e-2)
