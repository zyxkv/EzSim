import argparse

import numpy as np
import torch

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = ezsim.Scene(
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            constraint_solver=ezsim.constraint_solver.Newton,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(
        ezsim.morphs.Plane(),
    )
    robot = scene.add_entity(
        ezsim.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0, 0.4),
        ),
    )
    ########################## build ##########################
    n_envs = 8
    scene.build(n_envs=n_envs)

    ########################## domain randomization ##########################
    robot.set_friction_ratio(
        friction_ratio=0.5 + torch.rand(scene.n_envs, robot.n_links),
        links_idx_local=np.arange(0, robot.n_links),
    )

    # set mass of a single link
    link = robot.get_link("RR_thigh")
    rigid = scene.sim.rigid_solver
    ori_mass = rigid.links_info.inertial_mass.to_numpy()
    print("original mass", link.get_mass(), ori_mass)
    link.set_mass(1)
    new_mass = rigid.links_info.inertial_mass.to_numpy()
    print("diff mass", new_mass - ori_mass)

    robot.set_mass_shift(
        mass_shift=-0.5 + torch.rand(scene.n_envs, robot.n_links),
        links_idx_local=np.arange(0, robot.n_links),
    )
    robot.set_COM_shift(
        com_shift=-0.05 + 0.1 * torch.rand(scene.n_envs, robot.n_links, 3),
        links_idx_local=np.arange(0, robot.n_links),
    )
    aabb = robot.get_AABB()
    joints_name = (
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    )
    motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

    robot.set_dofs_kp(np.full(12, 20), motors_dof_idx)
    robot.set_dofs_kv(np.full(12, 1), motors_dof_idx)
    default_dof_pos = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
    default_dof_pos = np.tile(default_dof_pos, (n_envs, 1))
    robot.control_dofs_position(default_dof_pos, motors_dof_idx)

    scene.step()


if __name__ == "__main__":
    main()
