import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh

import ezsim
import ezsim.utils.geom as gu
from ezsim.utils.misc import get_assets_dir, tensor_to_array

from .utils import (
    assert_allclose,
    build_ezsim_sim,
    build_mujoco_sim,
    check_mujoco_data_consistency,
    check_mujoco_model_consistency,
    get_hf_assets,
    init_simulators,
    simulate_and_check_mujoco_consistency,
)


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_name = f"{model_name}.urdf" if mjcf.tag == "robot" else f"{model_name}.xml"
    file_path = str(tmp_path / file_name)
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.fixture(scope="session")
def box_plan():
    """Generate an MJCF model for a box on a plane."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box_body = ET.SubElement(worldbody, "body", name="box", pos="0. 0. 0.3")
    ET.SubElement(box_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box_body, "joint", name="root", type="free")
    return mjcf


@pytest.fixture(scope="session")
def mimic_hinges():
    mjcf = ET.Element("mujoco", model="mimic_hinges")
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = ET.SubElement(worldbody, "body", name="parent", pos="0 0 1.0")
    child1 = ET.SubElement(parent, "body", name="child1", pos="0.5 0 0")
    ET.SubElement(child1, "geom", type="capsule", size="0.05 0.2", rgba="0.9 0.1 0.1 1")
    ET.SubElement(child1, "joint", type="hinge", name="joint1", axis="0 1 0", range="-45 45")
    child2 = ET.SubElement(parent, "body", name="child2", pos="0 0.5 0")
    ET.SubElement(child2, "geom", type="capsule", size="0.05 0.2", rgba="0.1 0.1 0.9 1")
    ET.SubElement(child2, "joint", type="hinge", name="joint2", axis="0 1 0", range="-45 45")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(equality, "joint", name="joint_equality", joint1="joint1", joint2="joint2")
    return mjcf


@pytest.fixture(scope="session")
def box_box():
    """Generate an MJCF model for two boxes."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.2")
    ET.SubElement(box1_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.", rgba="0 1 0 0.4")
    ET.SubElement(box1_body, "joint", name="root1", type="free")
    box2_body = ET.SubElement(worldbody, "body", name="box2", pos="0. 0. 0.8")
    ET.SubElement(box2_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.", rgba="0 0 1 0.4")
    ET.SubElement(box2_body, "joint", name="root2", type="free")
    return mjcf


@pytest.fixture
def collision_edge_cases(asset_tmp_path, mode):
    assets = {}
    for i, box_size in enumerate(((0.8, 0.8, 0.04), (0.04, 0.04, 0.005))):
        tmesh = trimesh.creation.box(extents=np.array(box_size) * 2)
        mesh_path = str(asset_tmp_path / f"box{i}.obj")
        tmesh.export(mesh_path, file_type="obj")
        assets[f"box{i}"] = mesh_path

    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.005")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")

    asset = ET.SubElement(mjcf, "asset")
    for name, mesh_path in assets.items():
        ET.SubElement(asset, "mesh", name=name, refpos="0 0 0", refquat="1 0 0 0", file=mesh_path)

    worldbody = ET.SubElement(mjcf, "worldbody")

    if mode == 0:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 1:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 2:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="box", size="0.04 0.04 0.005", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 3:
        ET.SubElement(worldbody, "geom", type="box", size="0.8 0.8 0.04", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 4:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    elif mode == 5:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 6:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="-0.758 -0.758 1.1")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 7:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758  0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.758 -0.758 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.758  0.758 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    elif mode == 8:
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762  0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos=" 0.762 -0.762 0.", rgba="0 1 0 0.4")
        ET.SubElement(worldbody, "geom", type="mesh", mesh="box1", pos="-0.762  0.762 0.", rgba="0 1 0 0.4")
        box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0. 0. 0.7")
        ET.SubElement(box1_body, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 0 1 0.4")
    else:
        raise ValueError("Invalid mode")

    ET.SubElement(box1_body, "joint", name="root", type="free")

    return mjcf


@pytest.fixture(scope="session")
def two_aligned_hinges():
    mjcf = ET.Element("mujoco", model="two_aligned_hinges")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body0")
    ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link0, "joint", type="hinge", name="joint0", axis="0 0 1")
    link1 = ET.SubElement(link0, "body", name="body1", pos="0.5 0 0")
    ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1")
    return mjcf


def _build_chain_capsule_hinge(asset_tmp_path, enable_mesh):
    if enable_mesh:
        mesh_path = str(asset_tmp_path / "capsule.obj")
        tmesh = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        tmesh.apply_transform(np.diag([0.05, 0.05, 0.25, 1]))
        tmesh.export(mesh_path, file_type="obj")

    mjcf = ET.Element("mujoco", model="two_stick_robot")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    if enable_mesh:
        asset = ET.SubElement(mjcf, "asset")
        ET.SubElement(asset, "mesh", name="capsule", refpos="0 0 -0.25", refquat="0.707 0 -0.707 0", file=mesh_path)
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body1", pos="0.1 0.2 0.0", quat="0.707 0 0.707 0")
    if enable_mesh:
        ET.SubElement(link0, "geom", type="mesh", mesh="capsule", rgba="0 0 1 0.3")
    else:
        ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05", rgba="0 0 1 0.3")
    link1 = ET.SubElement(link0, "body", name="body2", pos="0.5 0.2 0.0", quat="0.92388 0 0 0.38268")
    if enable_mesh:
        ET.SubElement(link1, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1", pos="0.0 0.0 0.0")
    link2 = ET.SubElement(link1, "body", name="body3", pos="0.5 0.2 0.0", quat="0.92388 0 0.38268 0.0")
    if enable_mesh:
        ET.SubElement(link2, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link2, "joint", type="hinge", name="joint2", axis="0 1 0")
    return mjcf


@pytest.fixture(scope="session")
def chain_capsule_hinge_mesh(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=True)


@pytest.fixture(scope="session")
def chain_capsule_hinge_capsule(asset_tmp_path):
    return _build_chain_capsule_hinge(asset_tmp_path, enable_mesh=False)


def _build_multi_pendulum(n):
    """Generate an URDF model of a multi-link pendulum with n segments."""
    urdf = ET.Element("robot", name="multi_pendulum")

    # Base link
    ET.SubElement(urdf, "link", name="base")

    parent_link = "base"
    for i in range(n):
        # Continuous joint between parent and this arm
        joint = ET.SubElement(urdf, "joint", name=f"PendulumJoint_{i}", type="continuous")
        ET.SubElement(joint, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=f"PendulumArm_{i}")
        ET.SubElement(joint, "limit", effort=str(100.0 * (n - i)), velocity="30.0")

        # Arm link
        arm = ET.SubElement(urdf, "link", name=f"PendulumArm_{i}")
        visual = ET.SubElement(arm, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.5", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size="0.01 0.01 1.0")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(arm, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="0.0")
        ET.SubElement(inertial, "inertia", ixx="0.0", ixy="0.0", ixz="0.0", iyy="0.0", iyz="0.0", izz="0.0")

        # Fixed joint to the mass
        joint2 = ET.SubElement(urdf, "joint", name=f"PendulumMassJoint_{i}", type="fixed")
        ET.SubElement(joint2, "origin", xyz="0.0 0.0 1.0", rpy="0.0 0.0 0.0")
        ET.SubElement(joint2, "parent", link=f"PendulumArm_{i}")
        ET.SubElement(joint2, "child", link=f"PendulumMass_{i}")

        # Mass link
        mass = ET.SubElement(urdf, "link", name=f"PendulumMass_{i}")
        visual = ET.SubElement(mass, "visual")
        ET.SubElement(visual, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "sphere", radius="0.06")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0.0 0.0 1.0 1.0")
        inertial = ET.SubElement(mass, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="1e-12", ixy="0.0", ixz="0.0", iyy="1e-12", iyz="0.0", izz="1e-12")

        parent_link = f"PendulumMass_{i}"

    return urdf


@pytest.fixture(scope="session")
def pendulum(asset_tmp_path):
    return _build_multi_pendulum(n=1)


@pytest.fixture(scope="session")
def double_pendulum(asset_tmp_path):
    return _build_multi_pendulum(n=2)


@pytest.fixture(scope="session")
def double_ball_pendulum():
    mjcf = ET.Element("mujoco", model="double_ball_pendulum")

    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.1", damping="0.5")

    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base", pos="-0.02 0.0 0.0")
    ET.SubElement(base, "joint", name="joint1", type="ball")
    ET.SubElement(
        base, "geom", name="link1_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.5", rgba="0.8 0.2 0.2 1.0"
    )
    link2 = ET.SubElement(base, "body", name="link2", pos="0 0 0.5")
    ET.SubElement(link2, "joint", name="joint2", type="ball")
    ET.SubElement(
        link2, "geom", name="link2_geom", type="capsule", size="0.02", fromto="0 0 0 0 0 0.3", rgba="0.2 0.8 0.2 1.0"
    )
    ee = ET.SubElement(link2, "body", name="end_effector", pos="0 0 0.3")
    ET.SubElement(ee, "geom", name="ee_geom", type="sphere", size="0.02", density="200", rgba="1.0 0.8 0.2 1.0")

    return mjcf


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_box_plane_dynamics(ezsim_sim, mj_sim, tol):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_simple_kinematic_chain(ezsim_sim, mj_sim, tol):
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, num_steps=200, tol=tol)


# Disable Genesis multi-contact because it relies on discretized geometry unlike Mujoco
@pytest.mark.required
@pytest.mark.multi_contact(False)
@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize(
    "ezsim_solver",
    [
        ezsim.constraint_solver.CG,
        # ezsim.constraint_solver.Newton,  # FIXME: This test is not passing because collision detection is too sensitive
    ],
)
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_walker(ezsim_sim, mj_sim, gjk_collision, tol):
    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    (ezsim_robot,) = ezsim_sim.entities
    qpos = np.zeros((ezsim_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(ezsim_robot.n_dofs) * 0.2

    # Cannot simulate any longer because collision detection is very sensitive
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, qpos, qvel, num_steps=90, tol=tol)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_equality_joint(ezsim_sim, mj_sim, ezsim_solver):
    # there is an equality constraint
    assert ezsim_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    # Note that it is impossible to be more accurate than this because of the inherent stiffness of the problem.
    tol = 2e-8 if ezsim_solver == ezsim.constraint_solver.Newton else 1e-8
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    # check if the two joints are equal
    ezsim_qpos = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(ezsim_qpos[0], ezsim_qpos[1], tol=tol)


@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_equality_weld(ezsim_sim, mj_sim, ezsim_solver):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    ezsim_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    for entity in ezsim_sim.entities:
        for equality in entity.equalities:
            equality.set_sol_params(sol_params)
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    assert ezsim_sim.rigid_solver.n_equalities == 1
    np.random.seed(0)
    qpos = np.random.rand(ezsim_sim.rigid_solver.n_qs) * 0.1

    # Note that it is impossible to be more accurate than this because of the inherent stiffness of the problem.
    # The pose difference between Mujoco and Genesis (resulting from using quaternion instead of rotation matrices to
    # apply transform internally) is about 1e-15. This is fine and not surprising as it is consistent with machine
    # precision. These rounding errors are then amplified by 1e8 when computing the forces resulting from the kinematic
    # constraints. The constraints could be made softer by changing its impede parameters.
    tol = 1e-7 if ezsim_solver == ezsim.constraint_solver.Newton else 2e-5
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, qpos, num_steps=300, tol=tol)


def test_dynamic_weld(show_viewer, tol):
    scene = ezsim.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=ezsim.surfaces.Plastic(color=(1, 0, 0)),
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/universal_robots_ur5e/ur5e.xml",
        ),
    )
    scene.build(n_envs=4, env_spacing=(3.0, 3.0))

    end_effector = robot.get_link("ee_virtual_link")

    # Compute up and down robot configurations
    ee_pos_up = np.array((0.65, 0.0, 0.5), dtype=ezsim.np_float)
    ee_pos_down = np.array((0.65, 0.0, 0.15), dtype=ezsim.np_float)
    qpos_up = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_up, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=ezsim.np_float), (4, 1)),
    )
    qpos_down = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_down, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=ezsim.np_float), (4, 1)),
    )

    # move to pre-grasp pose
    robot.control_dofs_position(qpos_up)
    for i in range(120):
        scene.step()

    # reach
    robot.control_dofs_position(qpos_down)
    for i in range(70):
        scene.step()

    # add weld constraint and move back up
    scene.sim.rigid_solver.add_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1, 2))
    robot.control_dofs_position(qpos_up)
    for i in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), cube.get_quat()
    assert_allclose(torch.diff(cubes_quat, dim=0), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)

    # drop
    scene.sim.rigid_solver.delete_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1))
    for i in range(110):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), cube.get_quat()
    assert_allclose(torch.diff(cubes_quat, dim=0), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 3]], dim=0), 0.0, tol=1e-2)
    assert_allclose(cubes_pos[2] - cubes_pos[0], ee_pos_up - ee_pos_down, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/one_ball_joint.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_one_ball_joint(ezsim_sim, mj_sim, tol):
    # FIXME: Mujoco is detecting collision for some reason...
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, num_steps=600, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/rope_ball.xml", "xml/rope_hinge.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_rope_ball(ezsim_sim, mj_sim, ezsim_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    ezsim_sim.rigid_solver.set_dofs_position(ezsim_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(ezsim_sim, mj_sim, tol=tol)
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, num_steps=300, tol=5e-9)


@pytest.mark.required
@pytest.mark.multi_contact(False)
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_urdf_rope(
    ezsim_solver,
    ezsim_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    dof_damping,
    show_viewer,
):
    asset_path = get_hf_assets(pattern="linear_deformable.urdf")
    xml_path = os.path.join(asset_path, "linear_deformable.urdf")

    mj_sim = build_mujoco_sim(
        xml_path,
        ezsim_solver,
        ezsim_integrator,
        merge_fixed_links,
        multi_contact,
        adjacent_collision,
        dof_damping,
        gjk_collision,
    )
    ezsim_sim = build_ezsim_sim(
        xml_path,
        ezsim_solver,
        ezsim_integrator,
        merge_fixed_links,
        multi_contact,
        mujoco_compatibility,
        adjacent_collision,
        gjk_collision,
        show_viewer,
        mj_sim,
    )

    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    ezsim_sim.rigid_solver.set_global_sol_params(sol_params)
    mj_sim.model.jnt_solref[:, 0] = sol_params[0]
    mj_sim.model.geom_solref[:, 0] = sol_params[0]
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    # FIXME: Tolerance must be very large due to small masses and compounding of errors over long kinematic chains
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.mujoco_compatibility(True)
@pytest.mark.multi_contact(False)  # FIXME: Mujoco has errors with multi-contact, so this test is disabled
@pytest.mark.parametrize("xml_path", ["xml/tet_tet.xml", "xml/tet_ball.xml", "xml/tet_capsule.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_tet_primitive_shapes(ezsim_sim, mj_sim, ezsim_solver, xml_path, tol):
    # Make sure it is possible to set the configuration vector without failure
    ezsim_sim.rigid_solver.set_dofs_position(ezsim_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(ezsim_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error
    tol = 1e-6 if xml_path == "xml/tet_tet.xml" else 2e-8
    simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, num_steps=1000, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_aligned_hinges"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
def test_link_velocity(ezsim_sim, tol):
    # Check the velocity for a few "easy" special cases
    init_simulators(ezsim_sim, qvel=np.array([0.0, 1.0]))
    assert_allclose(ezsim_sim.rigid_solver.links_state.cd_vel.to_numpy(), 0, tol=tol)

    init_simulators(ezsim_sim, qvel=np.array([1.0, 0.0]))
    cvel_0, cvel_1 = ezsim_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, np.array([0.0, 0.5, 0.0]), tol=tol)
    assert_allclose(cvel_1, np.array([0.0, 0.5, 0.0]), tol=tol)

    init_simulators(ezsim_sim, qpos=np.array([0.0, np.pi / 2.0]), qvel=np.array([0.0, 1.2]))
    COM = ezsim_sim.rigid_solver.links_state.COM[0, 0]
    assert_allclose(COM, np.array([0.375, 0.125, 0.0]), tol=tol)
    xanchor = ezsim_sim.rigid_solver.joints_state.xanchor[1, 0]
    assert_allclose(xanchor, np.array([0.5, 0.0, 0.0]), tol=tol)
    cvel_0, cvel_1 = ezsim_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, 0, tol=tol)
    assert_allclose(cvel_1, np.array([-1.2 * (0.125 - 0.0), 1.2 * (0.375 - 0.5), 0.0]), tol=tol)

    # Check that the velocity is valid for a random configuration
    init_simulators(ezsim_sim, qpos=np.array([-0.7, 0.2]), qvel=np.array([3.0, 13.0]))
    xanchor = ezsim_sim.rigid_solver.joints_state.xanchor[1, 0]
    theta_0, theta_1 = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(xanchor[0], 0.5 * np.cos(theta_0), tol=tol)
    assert_allclose(xanchor[1], 0.5 * np.sin(theta_0), tol=tol)
    COM = ezsim_sim.rigid_solver.links_state.COM[0, 0]
    COM_0 = np.array([0.25 * np.cos(theta_0), 0.25 * np.sin(theta_0), 0.0])
    COM_1 = np.array(
        [
            0.5 * np.cos(theta_0) + 0.25 * np.cos(theta_0 + theta_1),
            0.5 * np.sin(theta_0) + 0.25 * np.sin(theta_0 + theta_1),
            0.0,
        ]
    )
    assert_allclose(COM, 0.5 * (COM_0 + COM_1), tol=tol)

    cvel_0, cvel_1 = ezsim_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    omega_0, omega_1 = ezsim_sim.rigid_solver.links_state.cd_ang.to_numpy()[:, 0, 2]
    assert_allclose(omega_0, 3.0, tol=tol)
    assert_allclose(omega_1 - omega_0, 13.0, tol=tol)
    cvel_0_ = omega_0 * np.array([-COM[1], COM[0], 0.0])
    assert_allclose(cvel_0, cvel_0_, tol=tol)
    cvel_1_ = cvel_0 + (omega_1 - omega_0) * np.array([xanchor[1] - COM[1], COM[0] - xanchor[0], 0.0])
    assert_allclose(cvel_1, cvel_1_, tol=tol)

    xpos_0, xpos_1 = ezsim_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    assert_allclose(xpos_0, 0.0, tol=tol)
    assert_allclose(xpos_1, xanchor, tol=tol)
    xvel_0, xvel_1 = ezsim_sim.rigid_solver.get_links_vel()
    assert_allclose(xvel_0, 0.0, tol=tol)
    xvel_1_ = omega_0 * np.array([-xpos_1[1], xpos_1[0], 0.0])
    assert_allclose(xvel_1, xvel_1_, tol=tol)
    civel_0, civel_1 = ezsim_sim.rigid_solver.get_links_vel(ref="link_com")
    civel_0_ = omega_0 * np.array([-COM_0[1], COM_0[0], 0.0])
    assert_allclose(civel_0, civel_0_, tol=tol)
    civel_1_ = omega_0 * np.array([-COM_1[1], COM_1[0], 0.0]) + (omega_1 - omega_0) * np.array(
        [xanchor[1] - COM_1[1], COM_1[0] - xanchor[0], 0.0]
    )
    assert_allclose(civel_1, civel_1_, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
def test_pendulum_links_acc(ezsim_sim, tol):
    pendulum = ezsim_sim.entities[0]
    g = ezsim_sim.rigid_solver._gravity[0][2]

    # Make sure that the linear and angular acceleration matches expectation
    theta = np.random.rand()
    theta_dot = np.random.rand()
    pendulum.set_qpos([theta])
    pendulum.set_dofs_velocity([theta_dot])
    for _ in range(100):
        # Backup state before integration
        theta = float(ezsim_sim.rigid_solver.qpos.to_numpy())
        theta_dot = float(ezsim_sim.rigid_solver.dofs_state.vel.to_numpy())

        # Run one simulation step
        ezsim_sim.scene.step()

        # Angular acceleration:
        # * acc_ang_x = - sin(theta) * g
        acc_ang = ezsim_sim.rigid_solver.get_links_acc_ang()
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[2], np.array([-np.sin(theta) * g, 0.0, 0.0]), tol=tol)
        # Linear spatial acceleration:
        # * acc_spatial_lin_y = sin(theta) * g
        acc_spatial_lin_world = ezsim_sim.rigid_solver.links_state.cacc_lin.to_numpy()
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), np.sin(theta)],
                [0.0, -np.sin(theta), np.cos(theta)],
            ]
        )
        acc_spatial_lin_local = R @ acc_spatial_lin_world[2, 0]
        assert_allclose(acc_spatial_lin_local, np.array([0.0, np.sin(theta) * g, 0.0]), tol=tol)
        # Linear true acceleration:
        # * acc_classical_lin_y = sin(theta) * g (tangential angular acceleration effect)
        # * acc_classical_lin_z = - theta_dot ** 2  (radial centripedal effect)
        acc_classical_lin_world = tensor_to_array(ezsim_sim.rigid_solver.get_links_acc(mimick_imu=False))
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = R @ acc_classical_lin_world[2]
        assert_allclose(acc_classical_lin_local, np.array([0.0, np.sin(theta) * g, -(theta_dot**2)]), tol=tol)
        # IMU accelerometer data:
        # * acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
        acc_imu = ezsim_sim.rigid_solver.get_links_acc(mimick_imu=True)[2]
        assert_allclose(acc_imu, np.array([0.0, 0.0, -(theta_dot**2) - np.cos(theta) * g]), tol=tol)

    # Hold the pendulum straight using PD controller and check again
    pendulum.set_dofs_kp([4000.0])
    pendulum.set_dofs_kv([100.0])
    pendulum.control_dofs_position([0.5 * np.pi])
    for _ in range(400):
        ezsim_sim.scene.step()
    acc_classical_lin_world = ezsim_sim.rigid_solver.get_links_acc(mimick_imu=False)
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["double_pendulum"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
def test_double_pendulum_links_acc(ezsim_sim, tol):
    robot = ezsim_sim.entities[0]

    # Make sure that the linear and angular acceleration matches expectation
    qpos = np.random.rand(2)
    qvel = np.random.rand(2)
    robot.set_qpos(qpos)
    robot.set_dofs_velocity(qvel)
    for _ in range(100):
        # Backup state before integration
        theta = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
        theta_dot = ezsim_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

        # Run one simulation step
        ezsim_sim.scene.step()

        # Backup acceleration before integration
        theta_ddot = ezsim_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]

        # Angular acceleration
        acc_ang = tensor_to_array(ezsim_sim.rigid_solver.get_links_acc_ang())
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[1], [theta_ddot[0], 0.0, 0.0], tol=tol)
        assert_allclose(acc_ang[-1], [theta_ddot[0] + theta_ddot[1], 0.0, 0.0], tol=tol)

        # Linear spatial acceleration
        cacc_spatial_lin_world = ezsim_sim.rigid_solver.links_state.cacc_lin.to_numpy()[[0, 2, 4], 0]
        com = ezsim_sim.rigid_solver.links_state.COM.to_numpy()[-1, 0]
        pos = ezsim_sim.rigid_solver.links_state.pos.to_numpy()[[0, 2, 4], 0]
        assert_allclose(cacc_spatial_lin_world[1], np.cross(acc_ang[2], com), tol=tol)
        acc_spatial_lin_world = cacc_spatial_lin_world + np.cross(acc_ang[[0, 2, 4]], pos - com)
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        theta_world = theta.cumsum()
        R = np.array(
            [
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                [np.zeros_like(theta), np.cos(theta_world), np.sin(theta_world)],
                [np.zeros_like(theta), -np.sin(theta_world), np.cos(theta_world)],
            ]
        )
        acc_spatial_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_spatial_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_spatial_lin_local[0], np.array([0.0, -theta_ddot[0], 0.0]), tol=tol)
        assert_allclose(
            acc_spatial_lin_local[1],
            R[..., 1] @ (R[..., 0].T @ np.array([0.0, -theta_ddot[0], theta_dot[0] * theta_dot[1]]))
            + np.array([0.0, -theta_ddot.sum(), 0.0]),
            tol=tol,
        )

        # Linear true acceleration
        acc_classical_lin_world = tensor_to_array(ezsim_sim.rigid_solver.get_links_acc(mimick_imu=False)[[0, 2, 4]])
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_classical_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_classical_lin_local[0], np.array([0.0, -theta_ddot[0], -theta_dot[0] ** 2]), tol=tol)
        assert_allclose(
            acc_classical_lin_local[1],
            R[..., 1] @ acc_classical_lin_world[1] + np.array([0.0, -theta_ddot.sum(), -theta_dot.sum() ** 2]),
            tol=tol,
        )

    # Hold the double pendulum straight using PD controller and check again
    robot.set_dofs_kp([6000.0, 4000.0])
    robot.set_dofs_kv([200.0, 150.0])
    robot.control_dofs_position([0.5 * np.pi, 0.0])
    for _ in range(900):
        ezsim_sim.scene.step()
    acc_classical_lin_world = ezsim_sim.rigid_solver.get_links_acc(mimick_imu=False)
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG, ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.implicitfast, ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_box_box_dynamics(ezsim_sim):
    (ezsim_robot,) = ezsim_sim.entities
    for _ in range(20):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.65 + 0.1 * np.random.rand()])
        cube2_quat = gu.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]),
        )
        init_simulators(ezsim_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(110):
            ezsim_sim.scene.step()
            if i > 100:
                qvel = ezsim_robot.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=1e-2)

        qpos = ezsim_robot.get_dofs_position()
        assert_allclose(qpos[8], 0.6, atol=2e-3)


@pytest.mark.parametrize(
    "box_box_detection, gjk_collision, dynamics",
    [
        (True, False, False),
        (False, False, False),
        (False, False, True),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("backend", [ezsim.cpu])  # TODO: Cannot afford GPU test for this one
def test_many_boxes_dynamics(box_box_detection, gjk_collision, dynamics, show_viewer):
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            box_box_detection=box_box_detection,
            max_collision_pairs=1000,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    for n in range(5**3):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        scene.add_entity(
            ezsim.morphs.Box(
                pos=(i * 1.01, j * 1.01, k * 1.01 + 0.5),
                size=(1.0, 1.0, 1.0),
            ),
            surface=ezsim.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )
    scene.build()

    if dynamics:
        for entity in scene.entities[1:]:
            entity.set_dofs_velocity(4.0 * np.random.rand(6))
    num_steps = 650 if dynamics else 150
    for i in range(num_steps):
        scene.step()
        if i > num_steps - 50:
            qvel = scene.rigid_solver.get_dofs_velocity().reshape((6, -1))
            # Checking the average velocity because is always one cube moving depending on the machine.
            assert_allclose(torch.linalg.norm(qvel, dim=0).mean(), 0, atol=0.05)

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position()
        if dynamics:
            assert qpos[:2].norm() < 20.0
            assert qpos[2] < 5.0
        else:
            qpos0 = np.array((i * 1.01, j * 1.01, k * 1.01 + 0.5))
            assert_allclose(qpos[:3], qpos0, atol=0.05)
            assert_allclose(qpos[3:], 0, atol=0.03)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/franka_emika_panda/panda.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_robot_kinematics(ezsim_sim, mj_sim, tol):
    # Disable all constraints and actuation
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    ezsim_sim.rigid_solver.dofs_state.ctrl_mode.fill(ezsim.CTRL_MODE.FORCE)
    ezsim_sim.rigid_solver._enable_collision = False
    ezsim_sim.rigid_solver._enable_joint_limit = False
    ezsim_sim.rigid_solver._disable_constraint = True

    check_mujoco_model_consistency(ezsim_sim, mj_sim, tol=tol)

    (ezsim_robot,) = ezsim_sim.entities
    dof_bounds = ezsim_sim.rigid_solver.dofs_info.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(ezsim_robot.n_qs)
        init_simulators(ezsim_sim, mj_sim, qpos)
        check_mujoco_data_consistency(ezsim_sim, mj_sim, tol=tol)


@pytest.mark.required
def test_robot_scaling(show_viewer, tol):
    mass = None
    links_pos = None
    for scale in (0.5, 1.0, 2.0):
        scene = ezsim.Scene(
            sim_options=ezsim.options.SimOptions(
                gravity=(0, 0, -10.0),
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        robot = scene.add_entity(
            ezsim.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                scale=scale,
            ),
        )
        scene.build()

        mass_ = robot.get_mass() / scale**3
        if mass is None:
            mass = mass_
        assert_allclose(mass, mass_, tol=tol)

        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        qpos = dofs_lower_bound
        robot.set_dofs_position(qpos)

        links_pos_ = robot.get_links_pos() / scale
        if links_pos is None:
            links_pos = links_pos_
        assert_allclose(links_pos, links_pos_, tol=tol)

        scene.step()
        qf_passive = scene.rigid_solver.dofs_state.qf_passive.to_numpy()
        assert_allclose(qf_passive, 0, tol=tol)


@pytest.mark.required
def test_info_batching(tol):
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            batch_dofs_info=True,
            batch_joints_info=True,
            batch_links_info=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build(n_envs=2)

    scene.step()
    qposs = robot.get_qpos()
    assert_allclose(qposs[0], qposs[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_pd_control(show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=ezsim.options.RigidOptions(
            batch_dofs_info=True,
            enable_self_collision=False,
            integrator=ezsim.integrator.approximate_implicitfast,
        ),
        # vis_options=ezsim.options.VisOptions(
        #     rendered_envs_idx=(1,),
        # ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2)

    MOTORS_POS_TARGET = torch.tensor(
        [0.6900, -0.1100, -0.7200, -2.7300, -0.1500, 2.6400, 0.8900, 0.0400, 0.0400],
        dtype=ezsim.tc_float,
        device=ezsim.device,
    )
    MOTORS_KP = torch.tensor(
        [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
        dtype=ezsim.tc_float,
        device=ezsim.device,
    )
    MOTORS_KD = torch.tensor(
        [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
        dtype=ezsim.tc_float,
        device=ezsim.device,
    )

    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)

    for i in range(1000):
        dofs_pos = robot.get_qpos(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) - MOTORS_KD * dofs_vel
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dofs_state.qf_applied.to_torch(device="cpu").T
        # dofs_torque = robot.get_dofs_control_force()
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("relative", [False, True])
def test_set_root_pose(relative, show_viewer, tol):
    ROBOT_POS_ZERO = (0.0, 0.4, 0.1)
    ROBOT_EULER_ZERO = (0.0, 0.0, 90.0)
    CUBE_POS_ZERO = (0.65, 0.0, 0.02)
    CUBE_EULER_ZERO = (0.0, 90.0, 0.0)

    scene = ezsim.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=ROBOT_POS_ZERO,
            euler=ROBOT_EULER_ZERO,
        ),
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=CUBE_POS_ZERO,
            euler=CUBE_EULER_ZERO,
        ),
    )
    scene.build()

    for _ in range(2):
        scene.reset()

        for entity, pos_zero, euler_zero in (
            (robot, ROBOT_POS_ZERO, ROBOT_EULER_ZERO),
            (cube, CUBE_POS_ZERO, CUBE_EULER_ZERO),
        ):
            pos_zero = torch.tensor(pos_zero, device="cpu", dtype=ezsim.tc_float)
            euler_zero = torch.deg2rad(torch.tensor(euler_zero, dtype=ezsim.tc_float))

            assert_allclose(entity.get_pos(), pos_zero, tol=tol)
            euler = gu.quat_to_xyz(entity.get_quat(), rpy=True)
            assert_allclose(euler, euler_zero, tol=5e-4)

            pos_delta = torch.rand(3, device="cpu", dtype=ezsim.tc_float)
            entity.set_pos(pos_delta, relative=relative)
            quat_delta = torch.rand(4, device="cpu", dtype=ezsim.tc_float)
            quat_delta /= torch.linalg.norm(quat_delta)
            entity.set_quat(quat_delta, relative=relative)

            pos_ref = pos_delta + pos_zero if relative else pos_delta
            assert_allclose(entity.get_pos(), pos_ref, tol=tol)
            euler = gu.quat_to_xyz(entity.get_quat(), rpy=True)
            quat_zero = gu.xyz_to_quat(euler_zero, rpy=True)
            if relative:
                quat_ref = gu.transform_quat_by_quat(quat_zero, quat_delta)
            else:
                quat_ref = quat_delta
            euler_ref = gu.quat_to_xyz(quat_ref, rpy=True)
            assert_allclose(euler, euler_ref, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs, batched", [(0, False), (3, True)])
def test_set_sol_params(n_envs, batched, tol):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        rigid_options=ezsim.options.RigidOptions(
            batch_joints_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.4, 0.1),
            euler=(0, 0, 90),
        ),
    )
    scene.build(n_envs=2)
    assert scene.sim._substep_dt == 0.01

    for objs, batched in ((robot.joints, batched), (robot.geoms, False), (robot.equalities, True)):
        for obj in objs:
            sol_params = obj.sol_params + 1.0
            obj.set_sol_params(sol_params)
            with pytest.raises(AssertionError):
                assert_allclose(obj.sol_params, sol_params, tol=tol)
            sol_params = np.zeros(((scene.n_envs,) if scene.n_envs > 0 and batched else ()) + (7,))
            obj.set_sol_params(sol_params)
            sol_params = np.tile(
                [2.0e-02, 0.0, 1e-4, 1e-4, 0.0, 1e-4, 1.0],
                ((scene.n_envs,) if scene.n_envs > 0 and batched else ()) + (1,),
            )
            assert_allclose(obj.sol_params, sol_params, tol=tol)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("xml_path", ["xml/humanoid.xml"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.Newton])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_stickman(ezsim_sim, mj_sim, tol):
    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(ezsim_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(ezsim_sim)

    # Run the simulation for a few steps
    qvel_norminf_all = []
    for i in range(6000):
        ezsim_sim.scene.step()
        if i > 4000:
            (ezsim_robot,) = ezsim_sim.entities
            qvel = ezsim_robot.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    np.testing.assert_array_less(torch.median(torch.stack(qvel_norminf_all, dim=0)).cpu(), 0.1)

    qpos = ezsim_robot.get_dofs_position()
    assert torch.linalg.norm(qpos[:2]) < 1.3
    body_z = ezsim_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z + ezsim.EPS)


def move_cube(use_suction, mode, show_viewer):
    # Add DoF armature to improve numerical stability if not using 'approximate_implicitfast' integrator.
    #
    # This is necessary because the first-order correction term involved in the implicit integration schemes
    # 'implicitfast' and 'Euler' are only able to stabilize each entity independently, from the forces that were
    # obtained from the instable accelerations. As a result, eveything is fine as long as the entities are not
    # interacting with each other, but it induces unrealistic motion otherwise. In this case, the acceleration of the
    # cube being lifted is based on the acceleration that the gripper would have without implicit damping.
    #
    # The only way to correct this would be to take into account the derivative of the Jacobian of the constraints in
    # the first-order correction term. Doing this is challenging and would significantly increase the computation cost.
    #
    # In practice, it is more common to just go for a higher order integrator such as RK4.
    if mode == 0:
        integrator = ezsim.integrator.approximate_implicitfast
        substeps = 1
        armature = 0.0
    elif mode == 1:
        integrator = ezsim.integrator.implicitfast
        substeps = 4
        armature = 0.0
    elif mode == 2:
        integrator = ezsim.integrator.Euler
        substeps = 1
        armature = 2.0

    # Create and build the scene
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
            substeps=substeps,
        ),
        rigid_options=ezsim.options.RigidOptions(
            box_box_detection=True,
            integrator=integrator,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.025),
        ),
        surface=ezsim.surfaces.Plastic(color=(1, 0, 0)),
    )
    cube_2 = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.4, 0.2, 0.025),
        ),
        surface=ezsim.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    franka.set_dofs_armature(franka.get_dofs_armature() + armature)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=300, resolution=0.05, max_retry=10)
    # execute the planned path
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(120):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.13]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(60):
        scene.step()

    # grasp
    if use_suction:
        link_cube = cube.get_link("box_baselink").idx
        link_franka = franka.get_link("hand").idx
        scene.sim.rigid_solver.add_weld_constraint(link_cube, link_franka)
    else:
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        for i in range(50):
            scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.2]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,
        resolution=0.05,
        max_retry=10,
        ee_link_name="hand",
        with_entity=cube,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(50):
        scene.step()

    # release
    if use_suction:
        scene.sim.rigid_solver.delete_weld_constraint(link_cube, link_franka)
    else:
        franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)

    for i in range(550):
        scene.step()
        if i > 550:
            qvel = cube.get_dofs_velocity()
            assert_allclose(qvel, 0, atol=0.02)

    qpos = cube.get_dofs_position()
    assert_allclose(qpos[2], 0.075, atol=2e-3)


@pytest.mark.parametrize(
    "mode, backend",
    [
        pytest.param(0, ezsim.cpu, marks=pytest.mark.required),
        pytest.param(1, ezsim.cpu),
        pytest.param(2, ezsim.cpu),
        pytest.param(0, ezsim.gpu),
        pytest.param(1, ezsim.gpu),
        pytest.param(2, ezsim.gpu),
    ],
)
def test_inverse_kinematics(mode, show_viewer):
    move_cube(use_suction=False, mode=mode, show_viewer=show_viewer)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_suction_cup(mode, show_viewer):
    move_cube(use_suction=True, mode=mode, show_viewer=show_viewer)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_path_planning_avoidance(n_envs, show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cubes = []
    for pos in (
        (-0.1, 0.2, 0.7),
        (0.0, 0.3, 0.8),
        (-0.1, -0.2, 0.7),
        (0.0, -0.3, 0.8),
        (0.3, 0.2, 0.6),
        (0.3, -0.2, 0.6),
        (0.3, 0.3, 0.7),
        (0.3, -0.3, 0.7),
    ):
        cube = scene.add_entity(
            ezsim.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=pos,
                fixed=True,
            ),
            surface=ezsim.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )
        cubes.append(cube)
    franka = scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=n_envs)

    hand = franka.get_link("hand")
    hand_pos_ref = torch.tensor([0.3, 0.25, 0.25], dtype=ezsim.tc_float, device=ezsim.device)
    hand_quat_ref = torch.tensor([0.3073, 0.5303, 0.7245, -0.2819], dtype=ezsim.tc_float, device=ezsim.device)
    if n_envs > 0:
        hand_pos_ref = hand_pos_ref.repeat((n_envs, 1))
        hand_quat_ref = hand_quat_ref.repeat((n_envs, 1))
    qpos = franka.inverse_kinematics(hand, pos=hand_pos_ref, quat=hand_quat_ref)
    qpos[..., -2:] = 0.04

    free_path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=300,
        resolution=0.05,
        ignore_collision=True,
    )
    assert_allclose(free_path[0], 0, tol=ezsim.EPS)
    assert_allclose(free_path[-1], qpos, tol=ezsim.EPS)

    qpos = franka.inverse_kinematics(hand, pos=hand_pos_ref, quat=hand_quat_ref)
    qpos[..., -2:] = 0.04
    avoidance_path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=300,
        ignore_collision=False,
        resolution=0.05,
        max_nodes=4000,
        max_retry=40,
    )
    assert_allclose(avoidance_path[0], 0.0, tol=ezsim.EPS)
    assert_allclose(avoidance_path[-1], qpos, tol=ezsim.EPS)

    for path, ignore_collision in ((free_path, False), (avoidance_path, True)):
        max_penetration = float("-inf")
        for waypoint in path:
            franka.set_qpos(waypoint)
            scene.visualizer.update()

            # Check if the cube is colliding with the robot
            scene.rigid_solver._func_forward_dynamics()
            scene.rigid_solver._func_constraint_force()
            for i in range(scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0]):
                contact_link_a = scene.rigid_solver.collider._collider_state.contact_data.link_a[i, 0]
                contact_link_b = scene.rigid_solver.collider._collider_state.contact_data.link_b[i, 0]
                penetration = scene.rigid_solver.collider._collider_state.contact_data.penetration[i, 0]
                if any(i_g in tuple(range(len(cubes))) for i_g in (contact_link_a, contact_link_b)):
                    max_penetration = max(max_penetration, penetration)
        args = (max_penetration, 5e-3)
        np.testing.assert_array_less(*(args if ignore_collision else args[::-1]))

        assert_allclose(hand_pos_ref, hand.get_pos(), tol=5e-4)
        hand_quat_diff = gu.transform_quat_by_quat(gu.inv_quat(hand_quat_ref), hand.get_quat())
        theta = 2 * torch.arctan2(torch.linalg.norm(hand_quat_diff[..., 1:]), torch.abs(hand_quat_diff[..., 0]))
        assert_allclose(theta, 0.0, tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_all_fixed(show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    scene.build()
    scene.step()

    assert_allclose(cube.get_pos(), 0, tol=ezsim.EPS)
    assert_allclose(cube.get_quat(), (1.0, 0.0, 0.0, 0.0), tol=ezsim.EPS)
    assert_allclose(cube.get_vel(), 0, tol=ezsim.EPS)
    assert_allclose(cube.get_ang(), 0, tol=ezsim.EPS)
    assert_allclose(scene.rigid_solver.get_links_acc(), 0, tol=ezsim.EPS)


@pytest.mark.required
def test_contact_forces(show_viewer, tol):
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            box_box_detection=True,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    franka = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        visualize_contact=True,
    )
    scene.build()

    cube_weight = scene.rigid_solver._gravity[0][2] * cube.get_mass()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(50):
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[0], [0.0, 0.0, -cube_weight], atol=1e-5)

    # grasp
    for i in range(20):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(200):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        scene.step()
    contact_forces = cube.get_links_net_contact_force()
    assert_allclose(contact_forces[0], [0.0, 0.0, -cube_weight], atol=5e-5)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["double_ball_pendulum"])
def test_apply_external_forces(xml_path, show_viewer):
    scene = ezsim.Scene(
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    robot = scene.add_entity(
        ezsim.morphs.MJCF(
            file=xml_path,
            quat=(1.0, 0, 1.0, 0),
        ),
    )
    scene.build()

    tol = 5e-3
    end_effector_link_idx = robot.links[-1].idx
    for step in range(801):
        ee_pos = scene.rigid_solver.get_links_pos([end_effector_link_idx])[0]
        if step == 0:
            assert_allclose(ee_pos, [0.8, 0.0, 0.02], tol=tol)
        elif step == 600:
            assert_allclose(ee_pos, [0.0, 0.0, 0.82], tol=tol)
        elif step == 800:
            assert_allclose(ee_pos, [-0.8 / math.sqrt(2), 0.8 / math.sqrt(2), 0.02], tol=tol)

        if step >= 600:
            force = np.array([[-5.0, 5.0, 0.0]])
        elif step >= 100:
            force = np.array([[0.0, 0.0, 10.0]])
        else:
            force = np.array([[0.0, 0.0, 0.0]])

        scene.rigid_solver.apply_links_external_force(force=force, links_idx=[end_effector_link_idx])
        scene.step()


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_mass_mat(show_viewer, tol):
    # Create and build the scene
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    franka1 = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    franka2 = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 2, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_nonconvex_collision(show_viewer):
    scene = ezsim.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    tank = scene.add_entity(
        ezsim.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
            convexify=False,
        ),
    )
    ball = scene.add_entity(
        ezsim.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.8),
        ),
        surface=ezsim.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
        visualize_contact=True,
    )
    scene.build()

    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    ball.set_dofs_velocity(np.random.rand(ball.n_dofs) * 0.8)
    for i in range(1800):
        scene.step()
        if i > 1700:
            qvel = scene.sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
            assert_allclose(qvel, 0, atol=0.65)


@pytest.mark.parametrize("convexify", [True, False])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_mesh_repair(convexify, show_viewer, gjk_collision):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=ezsim.options.RigidOptions(
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_assets(pattern="work_table.glb")
    table = scene.add_entity(
        ezsim.morphs.Mesh(
            file=f"{asset_path}/work_table.glb",
            pos=(0.4, 0.0, -0.54),
            fixed=True,
        ),
        vis_mode="collision",
    )
    asset_path = get_hf_assets(pattern="spoon.glb")
    obj = scene.add_entity(
        ezsim.morphs.Mesh(
            file=f"{asset_path}/spoon.glb",
            pos=(0.3, 0, 0.015),
            quat=(0.707, 0.707, 0, 0),
            convexify=convexify,
            scale=1.0,
        ),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    if convexify:
        assert all(geom.metadata["decomposed"] for geom in obj.geoms)

    # MPR collision detection is significantly less reliable than SDF in terms of penetration depth estimation.
    tol_pos = 0.05 if convexify else 1e-6
    tol_rot = 1.3 if convexify else 1e-4
    for i in range(400):
        scene.step()
        if i > 300:
            qvel = obj.get_dofs_velocity()
            assert_allclose(qvel[:3], 0, atol=tol_pos)
            assert_allclose(qvel[3:], 0, atol=tol_rot)
    qpos = obj.get_dofs_position()
    assert_allclose(qpos[:2], (0.3, 0.0), atol=2e-3)


# FIXME: GJK collision detection algorithm is failing on some platform.
@pytest.mark.required
@pytest.mark.parametrize("euler", [(90, 0, 90), (75, 15, 90)])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_convexify(euler, backend, show_viewer, gjk_collision):
    OBJ_OFFSET_X = 0.0  # 0.02
    OBJ_OFFSET_Y = 0.15

    # The test check that the volume difference is under a given threshold and
    # that convex decomposition is only used whenever it is necessary.
    # Then run a simulation to see if it explodes, i.e. objects are at reset inside tank.
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.004,
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        ezsim.morphs.URDF(
            file="urdf/blue_box/model.urdf",
            fixed=True,
            pos=(0.0, 1.0, 0.0),
        ),
        vis_mode="collision",
    )
    tank = scene.add_entity(
        ezsim.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            pos=(0.05, -0.1, 0.0),
            euler=euler,
            # coacd_options=ezsim.options.CoacdOptions(
            #     threshold=0.08,
            # ),
        ),
        vis_mode="collision",
    )
    objs = []
    for i, asset_name in enumerate(("mug_1", "donut_0", "cup_2", "apple_15")):
        asset_path = get_hf_assets(pattern=f"{asset_name}/*")
        obj = scene.add_entity(
            ezsim.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/output.xml",
                pos=(OBJ_OFFSET_X * (1.5 - i), OBJ_OFFSET_Y * (i - 1.5), 0.4),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        objs.append(obj)
    # cam = scene.add_camera(
    #     pos=(0.5, 0.0, 1.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     res=(500, 500),
    #     fov=60,
    #     spp=512,
    #     GUI=False,
    # )
    scene.build()
    ezsim_sim = scene.sim

    # Make sure that all the geometries in the scene are convex
    assert ezsim_sim.rigid_solver.geoms_info.is_convex.to_numpy().all()
    assert not ezsim_sim.rigid_solver.collider._collider_static_config.has_nonconvex_nonterrain

    # There should be only one geometry for the apple as it can be convexify without decomposition,
    # but for the others it is hard to tell... Let's use some reasonable guess.
    mug, donut, cup, apple = objs
    assert len(apple.geoms) == 1
    assert all(geom.metadata["decomposed"] for geom in donut.geoms) and 5 <= len(donut.geoms) <= 10
    assert all(geom.metadata["decomposed"] for geom in cup.geoms) and 5 <= len(cup.geoms) <= 20
    assert all(geom.metadata["decomposed"] for geom in mug.geoms) and 5 <= len(mug.geoms) <= 40
    assert all(geom.metadata["decomposed"] for geom in box.geoms) and 5 <= len(box.geoms) <= 20

    # Check resting conditions repeateadly rather not just once, for numerical robustness
    # cam.start_recording()
    qvel_norminf_all = []
    for i in range(1700):
        scene.step()
        # cam.render()
        if i > 1600:
            qvel = ezsim_sim.rigid_solver.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    np.testing.assert_array_less(torch.median(torch.stack(qvel_norminf_all, dim=0)).cpu(), 4.0)
    # cam.stop_recording(save_to_filename="video.mp4", fps=60)

    for obj in objs:
        qpos = obj.get_dofs_position().cpu()
        np.testing.assert_array_less(-0.1, qpos[2])
        np.testing.assert_array_less(qpos[2], 0.15)
        np.testing.assert_array_less(torch.linalg.norm(qpos[:2]), 0.5)

    # Check that the mug and donut are landing straight if the tank is horizontal.
    # The cup is tipping because it does not land flat due to convex decomposition error.
    if euler == (90, 0, 90):
        for i, obj in enumerate((mug, donut)):
            qpos = obj.get_dofs_position()
            assert_allclose(qpos[0], OBJ_OFFSET_X * (1.5 - i), atol=7e-3)
            assert_allclose(qpos[1], OBJ_OFFSET_Y * (i - 1.5), atol=5e-3)


@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("mode", range(9))
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_collision_edge_cases(ezsim_sim, mode, gjk_collision):
    qpos_0 = ezsim_sim.rigid_solver.get_dofs_position()
    for _ in range(200):
        ezsim_sim.scene.step()

    qvel = ezsim_sim.rigid_solver.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)
    qpos = ezsim_sim.rigid_solver.get_dofs_position()
    # When using GJK, tolerance should be slightly higher for mode 6, but it is still physically valid.
    atol = 1e-3 if gjk_collision == True and mode == 6 else 1e-4
    assert_allclose(qpos[[0, 1, 3, 4, 5]], qpos_0[[0, 1, 3, 4, 5]], atol=atol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_collision_plane_convex(show_viewer, tol):
    for morph in (
        ezsim.morphs.Plane(),
        ezsim.morphs.Box(
            pos=(0.5, 0.0, -0.5),
            size=(1.0, 1.0, 1.0),
            fixed=True,
        ),
    ):
        scene = ezsim.Scene(
            sim_options=ezsim.options.SimOptions(
                dt=0.001,
            ),
            viewer_options=ezsim.options.ViewerOptions(
                camera_pos=(1.0, -0.5, 0.5),
                camera_lookat=(0.5, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        scene.add_entity(morph)

        asset_path = get_hf_assets(pattern="image_0000_segmented.glb")
        asset = scene.add_entity(
            ezsim.morphs.Mesh(
                file=f"{asset_path}/image_0000_segmented.glb",
                scale=0.03196910891804585,
                pos=(0.45184245, 0.05020455, 0.02),
                quat=(0.51982231, 0.44427745, 0.49720965, 0.53402704),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )

        scene.build()

        for i in range(500):
            scene.step()
            if i > 400:
                qvel = asset.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=0.14)


@pytest.mark.xfail(reason="No reliable way to generate nan on all platforms.")
@pytest.mark.parametrize("mode", [3])
@pytest.mark.parametrize("model_name", ["collision_edge_cases"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_nan_reset(ezsim_sim, mode):
    for _ in range(200):
        ezsim_sim.scene.step()
        qvel = ezsim_sim.rigid_solver.get_dofs_velocity()
        if torch.isnan(qvel).any():
            break
    else:
        raise AssertionError

    ezsim_sim.scene.reset()
    for _ in range(5):
        ezsim_sim.scene.step()
    qvel = ezsim_sim.rigid_solver.get_dofs_velocity()
    assert not torch.isnan(qvel).any()


@pytest.mark.parametrize("backend", [ezsim.cpu, ezsim.gpu])
def test_terrain_generation(show_viewer):
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    terrain = scene.add_entity(
        morph=ezsim.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(6.0, 6.0),
            horizontal_scale=0.25,
            vertical_scale=0.005,
            subterrain_types=[
                ["flat_terrain", "random_uniform_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
            ],
        ),
    )
    ball = scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(1.0, 1.0, 1.0),
            radius=0.1,
        ),
    )
    scene.build(n_envs=225)

    ball.set_pos(torch.cartesian_prod(*(torch.linspace(1.0, 10.0, 15),) * 2, torch.tensor((0.6,))))
    for _ in range(400):
        scene.step()

    # Make sure that at least one ball is as minimum height, and some are signficantly higher
    height_field = terrain.geoms[0].metadata["height_field"]
    height_field_min = terrain.terrain_scale[1] * height_field.min()
    height_field_max = terrain.terrain_scale[1] * height_field.max()
    height_balls = ball.get_pos()[:, 2]
    height_balls_min = height_balls.min() - 0.1
    height_balls_max = height_balls.max() - 0.1
    assert_allclose(height_balls_min, height_field_min, atol=2e-3)
    assert height_balls_max - height_balls_min > 0.5 * (height_field_max - height_field_min)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_terrain_size(show_viewer, tol):
    scene_ref = ezsim.Scene(show_viewer=show_viewer)
    terrain_ref = scene_ref.add_entity(
        morph=ezsim.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]

    scene_test = ezsim.Scene(show_viewer=show_viewer)
    terrain_test = scene_test.add_entity(
        morph=ezsim.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            subterrain_parameters={"wave_terrain": {"amplitude": 0.2}},
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]

    assert_allclose((height_ref * 2.0), height_test, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("ezsim_solver", [ezsim.constraint_solver.CG])
@pytest.mark.parametrize("ezsim_integrator", [ezsim.integrator.Euler])
def test_jacobian(ezsim_sim, tol):
    pendulum = ezsim_sim.entities[0]
    angle = 0.7
    pendulum.set_qpos(np.array([angle], dtype=ezsim.np_float))
    ezsim_sim.scene.step()

    link = pendulum.get_link("PendulumArm_0")

    p_local = np.array([0.05, -0.02, 0.12], dtype=ezsim.np_float)
    J_o = tensor_to_array(pendulum.get_jacobian(link))
    J_p = tensor_to_array(pendulum.get_jacobian(link, p_local))

    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=ezsim.np_float,
    )
    r_world = Rx @ p_local
    r_cross = np.array(
        [
            [0, -r_world[2], r_world[1]],
            [r_world[2], 0, -r_world[0]],
            [-r_world[1], r_world[0], 0],
        ],
        dtype=ezsim.np_float,
    )

    lin_o, ang_o = J_o[:3, 0], J_o[3:, 0]
    lin_expected = lin_o - r_cross @ ang_o

    assert_allclose(J_p[3:, 0], ang_o, tol=tol)
    assert_allclose(J_p[:3, 0], lin_expected, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_urdf_parsing(show_viewer, tol):
    POS_OFFSET = 0.8
    WOLRD_QUAT = np.array([1.0, 1.0, -0.3, +0.3])
    DOOR_JOINT_DAMPING = 1.5

    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            # Must use GJK to make collision detection independent from the center of each geometry.
            # Note that it is also the case for MPR+SDF most of the time due to warm-start.
            use_gjk_collision=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_assets(pattern="microwave/*")
    entities = {}
    for i, (fixed, merge_fixed_links) in enumerate(
        ((False, False), (False, True), (True, False), (True, True)),
    ):
        entity = scene.add_entity(
            morph=ezsim.morphs.URDF(
                file=f"{asset_path}/microwave/microwave.urdf",
                fixed=fixed,
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, (i - 1.5) * POS_OFFSET, 0.0),
                quat=tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT)),
            ),
            vis_mode="collision",
        )
        entities[(fixed, merge_fixed_links)] = entity
    scene.build()

    # four microwaves have four different root_idx
    root_idx_all = [link.root_idx for link in scene.rigid_solver.links]
    assert len(set(root_idx_all)) == 4

    def _check_entity_positions(relative, tol):
        nonlocal entities
        AABB_all = []
        for key in ((False, False), (False, True), (True, False), (True, True)):
            AABB = np.array(
                [
                    [np.inf, np.inf, np.inf],
                    [-np.inf, -np.inf, -np.inf],
                ]
            )
            for geom in entities[key].geoms:
                AABB_i = geom.get_AABB()
                AABB[0] = np.minimum(AABB[0], AABB_i[0])
                AABB[1] = np.maximum(AABB[1], AABB_i[1])
            AABB_all.append(AABB)
        AABB_diff = np.diff(AABB_all, axis=0)
        if relative:
            AABB_diff[..., 1] -= POS_OFFSET
        assert_allclose(AABB_diff, 0.0, tol=tol)

    # Check that `set_pos` / `set_quat` applies the same transform in all cases
    for relative in (False, True):
        for key in ((False, False), (False, True), (True, False), (True, True)):
            entities[key].set_pos(np.array([0.5, 0.0, 0.0]), relative=relative)
            entities[key].set_quat(np.array([0.0, 0.0, 0.0, 1.0]), relative=relative)
        if show_viewer:
            scene.visualizer.update()
        _check_entity_positions(relative, tol=ezsim.EPS)

    # Check that `set_qpos` applies the same absolute transform in all cases
    door_angle = np.array([1.1])
    for i, key in enumerate(((False, False), (False, True))):
        qpos = np.concatenate(
            ((0.0, (i - 1.5) * POS_OFFSET, 0.0), tuple(WOLRD_QUAT / np.linalg.norm(WOLRD_QUAT)), door_angle)
        )
        entities[key].set_qpos(qpos)
    for i, key in enumerate(((True, False), (True, True))):
        entities[key].set_pos(np.array([0.0, 0.0, 0.0]), relative=True)
        entities[key].set_quat(np.array([1.0, 0.0, 0.0, 0.0]), relative=True)
        entities[key].set_qpos(door_angle)
    if show_viewer:
        scene.visualizer.update()
    _check_entity_positions(relative=True, tol=ezsim.EPS)

    # Add dof damping to stabilitze the physics
    for key in ((False, False), (False, True), (True, False), (True, True)):
        entities[key].set_dofs_damping(entities[key].get_dofs_damping() + DOOR_JOINT_DAMPING)

    # Make sure that the dynamics of the door is the same in all cases
    door_vel = np.array([-0.2])
    entities[(False, False)].set_dofs_velocity(door_vel, 6)
    entities[(False, True)].set_dofs_velocity(door_vel, 6)
    entities[(True, False)].set_dofs_velocity(door_vel)
    entities[(True, True)].set_dofs_velocity(door_vel)
    link_1 = entities[(True, True)].link_start
    for key in ((False, False), (False, True)):
        link_2 = entities[key].link_start
        scene.rigid_solver.add_weld_constraint(link_1, link_2)

    for i in range(2000):
        scene.step()
        door_pos_all = (
            entities[(False, False)].get_dofs_position(6),
            entities[(False, True)].get_dofs_position(6),
            entities[(True, False)].get_dofs_position(0),
            entities[(True, True)].get_dofs_position(0),
        )
        door_pos_diff = np.diff(torch.concatenate(door_pos_all))
        assert_allclose(door_pos_diff, 0, tol=5e-3)
    assert_allclose(scene.rigid_solver.dofs_state.vel.to_numpy(), 0.0, tol=1e-3)
    _check_entity_positions(relative=True, tol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_urdf_mimic(show_viewer, tol):
    # create and build the scene
    scene = ezsim.Scene(
        show_viewer=show_viewer,
    )
    hand = scene.add_entity(
        ezsim.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            fixed=True,
        ),
    )
    scene.build()
    assert scene.rigid_solver.n_equalities == 1

    qvel = scene.rigid_solver.dofs_state.vel.to_numpy()
    qvel[-1] = 1
    scene.rigid_solver.dofs_state.vel.from_numpy(qvel)
    for i in range(200):
        scene.step()

    ezsim_qpos = scene.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(ezsim_qpos[-1], ezsim_qpos[-2], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_gravity(show_viewer, tol):
    scene = ezsim.Scene(
        show_viewer=show_viewer,
    )

    sphere = scene.add_entity(ezsim.morphs.Sphere())
    scene.build(n_envs=2)

    scene.sim.set_gravity(torch.tensor([0.0, 0.0, -9.8]), envs_idx=0)
    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 9.8]), envs_idx=1)

    for _ in range(200):
        scene.step()

    first_pos = sphere.get_dofs_position()[0, 2]
    second_pos = sphere.get_dofs_position()[1, 2]

    assert_allclose(first_pos * -1, second_pos, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_scene_saver_franka(show_viewer, tol):
    scene1 = ezsim.Scene(
        show_viewer=show_viewer,
        profiling_options=ezsim.options.ProfilingOptions(
            show_FPS=False,
        ),
    )
    franka1 = scene1.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene1.build()

    dof_idx = [j.dofs_idx_local[0] for j in franka1.joints]

    franka1.set_dofs_kp(np.full(len(dof_idx), 3000), dof_idx)
    franka1.set_dofs_kv(np.full(len(dof_idx), 300), dof_idx)

    target_pose = np.array([0.3, -0.8, 0.4, -1.6, 0.5, 1.0, -0.6, 0.03, 0.03], dtype=float)
    franka1.control_dofs_position(target_pose, dof_idx)

    for _ in range(400):
        scene1.step()

    pose_ref = franka1.get_dofs_position(dof_idx)

    ckpt_path = Path(tempfile.gettempdir()) / "franka_unit.pkl"
    scene1.save_checkpoint(ckpt_path)

    scene2 = ezsim.Scene(show_viewer=show_viewer)
    franka2 = scene2.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene2.build()
    scene2.load_checkpoint(ckpt_path)

    pose_loaded = franka2.get_dofs_position(dof_idx)

    assert_allclose(pose_ref, pose_loaded, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_drone_hover_same_with_and_without_substeps(show_viewer, tol):
    base_rpm = 15000
    scene_ref = ezsim.Scene(
        show_viewer=show_viewer,
        sim_options=ezsim.options.SimOptions(
            dt=0.002,
            substeps=1,
        ),
    )
    drone_ref = scene_ref.add_entity(
        morph=ezsim.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_ref.build()

    for _ in range(2500):
        drone_ref.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_ref.step()

    pos_ref = drone_ref.get_dofs_position()

    scene_test = ezsim.Scene(
        show_viewer=show_viewer,
        sim_options=ezsim.options.SimOptions(
            dt=0.01,
            substeps=5,
        ),
    )
    drone_test = scene_test.add_entity(
        morph=ezsim.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_test.build()

    for _ in range(500):
        drone_test.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_test.step()

    pos_test = drone_test.get_dofs_position()

    assert_allclose(pos_ref, pos_test, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_drone_advanced(show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(ezsim.morphs.Plane())
    asset_path = get_hf_assets(pattern="drone_sus/*")
    drones = []
    for offset, merge_fixed_links in ((-0.3, False), (0.3, True)):
        drone = scene.add_entity(
            morph=ezsim.morphs.Drone(
                file=f"{asset_path}/drone_sus/drone_sus.urdf",
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, offset, 1.5),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        drones.append(drone)
    scene.build()

    for drone in drones:
        chain_dofs = range(6, drone.n_dofs)
        drone.set_dofs_armature(drone.get_dofs_armature(chain_dofs) + 1e-3, chain_dofs)

    # Wait for the drones to land on the ground and hold straight
    for i in range(400):
        for drone in drones:
            drone.set_propellels_rpm(torch.full((4,), 50000.0))
        scene.step()
        if i > 350:
            assert scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] == 2
            assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0, tol=2e-3)

    # Push the drones symmetrically and wait for them to collide
    drones[0].set_dofs_velocity([0.2], [1])
    drones[1].set_dofs_velocity([-0.2], [1])
    for i in range(150):
        for drone in drones:
            drone.set_propellels_rpm(torch.full((4,), 50000.0))
        scene.step()
        if scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] > 2:
            break
    else:
        raise AssertionError

    tol = 1e-2
    pos_1 = drones[0].get_pos()
    pos_2 = drones[1].get_pos()
    assert abs(pos_1[0] - pos_2[0]) < tol
    assert abs(pos_1[1] + pos_2[1]) < tol
    assert abs(pos_1[2] - pos_2[2]) < tol
    quat_1 = drones[0].get_quat()
    quat_2 = drones[1].get_quat()
    assert abs(quat_1[1] + quat_2[1]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol


@pytest.mark.parametrize(
    "n_envs, batched, backend",
    [
        (0, False, ezsim.cpu),
        (0, False, ezsim.gpu),
        (3, False, ezsim.cpu),
        # (3, True, ezsim.cpu),  # FIXME: Must refactor the unit test to support batching
    ],
)
def test_data_accessor(n_envs, batched, tol):
    # Create and build the scene
    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            batch_dofs_info=batched,
            batch_joints_info=batched,
            batch_links_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(ezsim.morphs.Plane())
    ezsim_robot = scene.add_entity(
        ezsim.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    ezsim_link = ezsim_robot.get_link("RR_thigh")
    scene.build(n_envs=n_envs)
    ezsim_s = scene.sim.rigid_solver

    # Initialize the simulation
    np.random.seed(0)
    dof_bounds = ezsim_s.dofs_info.limit.to_torch(device="cpu")
    dof_bounds[..., :2, :] = torch.tensor((-1.0, 1.0))
    dof_bounds[..., 2, :] = torch.tensor((0.7, 1.0))
    dof_bounds[..., 3:6, :] = torch.tensor((-np.pi / 2, np.pi / 2))
    for i in range(max(n_envs, 1)):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(ezsim_robot.n_dofs)
        ezsim_robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

    # Simulate for a while, until they collide with something
    for _ in range(400):
        scene.step()

        ezsim_n_contacts = ezsim_s.collider._collider_state.n_contacts.to_numpy()
        assert len(ezsim_n_contacts) == max(n_envs, 1)
        for as_tensor in (False, True):
            for to_torch in (False, True):
                contacts_info = ezsim_s.collider.get_contacts(as_tensor, to_torch)
                for value in contacts_info.values():
                    if n_envs > 0:
                        assert n_envs == len(value)
                    else:
                        assert ezsim_n_contacts[0] == len(value)
                        value = value[None] if as_tensor else (value,)

                    for i_b in range(n_envs):
                        n_contacts = ezsim_n_contacts[i_b]
                        if as_tensor:
                            assert isinstance(value, torch.Tensor if to_torch else np.ndarray)
                            if value.dtype in (ezsim.tc_int, ezsim.np_int):
                                assert (value[i_b, :n_contacts] != -1).all()
                                assert (value[i_b, n_contacts:] == -1).all()
                            else:
                                assert_allclose(value[i_b, n_contacts:], 0.0, tol=0)
                        else:
                            assert isinstance(value, (list, tuple))
                            assert value[i_b].shape[0] == n_contacts
                            if value[i_b].dtype in (ezsim.tc_int, ezsim.np_int):
                                assert (value[i_b] != -1).all()

        if (ezsim_n_contacts > 0).all():
            break
    else:
        assert False
    ezsim_s._func_forward_dynamics()
    ezsim_s._func_constraint_force()

    # ezsim_robot.get_contacts()

    # Make sure that all the robots ends up in the different state
    qposs = ezsim_robot.get_qpos()
    for i in range(n_envs - 1):
        with np.testing.assert_raises(AssertionError):
            assert_allclose(qposs[i], qposs[i + 1], tol=tol)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with random value -> Call 'Get'
    # * Compare first 'Get' ouput with field value
    # Then, for any possible combinations of row and column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # * Compare first 'Get' output with last 'Get' output
    # * Compare last 'Get' output with corresponding slice of non-masking 'Get' output
    def get_all_supported_masks(i):
        return (
            i,
            [i],
            slice(i, i + 1),
            range(i, i + 1),
            np.array([i], dtype=np.int32),
            torch.tensor([i], dtype=torch.int64),
            torch.tensor([i], dtype=ezsim.tc_int, device=ezsim.device),
        )

    def must_cast(value):
        return not (isinstance(value, torch.Tensor) and value.dtype == ezsim.tc_int and value.device == ezsim.device)

    for arg1_max, arg2_max, getter_or_spec, setter, field in (
        # SOLVER
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_pos, None, ezsim_s.links_state.pos),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_quat, None, ezsim_s.links_state.quat),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_vel, None, None),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_ang, None, ezsim_s.links_state.cd_ang),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_acc, None, None),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_root_COM, None, ezsim_s.links_state.COM),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_mass_shift, ezsim_s.set_links_mass_shift, ezsim_s.links_state.mass_shift),
        (ezsim_s.n_links, n_envs, ezsim_s.get_links_COM_shift, ezsim_s.set_links_COM_shift, ezsim_s.links_state.i_pos_shift),
        (ezsim_s.n_links, -1, ezsim_s.get_links_inertial_mass, ezsim_s.set_links_inertial_mass, ezsim_s.links_info.inertial_mass),
        (ezsim_s.n_links, -1, ezsim_s.get_links_invweight, ezsim_s.set_links_invweight, ezsim_s.links_info.invweight),
        (ezsim_s.n_dofs, n_envs, ezsim_s.get_dofs_control_force, ezsim_s.control_dofs_force, None),
        (ezsim_s.n_dofs, n_envs, ezsim_s.get_dofs_force, None, ezsim_s.dofs_state.force),
        (ezsim_s.n_dofs, n_envs, ezsim_s.get_dofs_velocity, ezsim_s.set_dofs_velocity, ezsim_s.dofs_state.vel),
        (ezsim_s.n_dofs, n_envs, ezsim_s.get_dofs_position, ezsim_s.set_dofs_position, ezsim_s.dofs_state.pos),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_force_range, ezsim_s.set_dofs_force_range, ezsim_s.dofs_info.force_range),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_limit, None, ezsim_s.dofs_info.limit),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_stiffness, None, ezsim_s.dofs_info.stiffness),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_invweight, None, ezsim_s.dofs_info.invweight),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_armature, ezsim_s.set_dofs_armature, ezsim_s.dofs_info.armature),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_damping, ezsim_s.set_dofs_damping, ezsim_s.dofs_info.damping),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_kp, ezsim_s.set_dofs_kp, ezsim_s.dofs_info.kp),
        (ezsim_s.n_dofs, -1, ezsim_s.get_dofs_kv, ezsim_s.set_dofs_kv, ezsim_s.dofs_info.kv),
        (ezsim_s.n_geoms, n_envs, ezsim_s.get_geoms_pos, None, ezsim_s.geoms_state.pos),
        (ezsim_s.n_geoms, n_envs, ezsim_s.get_geoms_friction_ratio, ezsim_s.set_geoms_friction_ratio,ezsim_s.geoms_state.friction_ratio,),
        (ezsim_s.n_geoms, -1, ezsim_s.get_geoms_friction, ezsim_s.set_geoms_friction, ezsim_s.geoms_info.friction),
        (ezsim_s.n_qs, n_envs, ezsim_s.get_qpos, ezsim_s.set_qpos, ezsim_s.qpos),
        # ROBOT
        (ezsim_robot.n_links, n_envs, ezsim_robot.get_links_pos, None, None),
        (ezsim_robot.n_links, n_envs, ezsim_robot.get_links_quat, None, None),
        (ezsim_robot.n_links, n_envs, ezsim_robot.get_links_vel, None, None),
        (ezsim_robot.n_links, n_envs, ezsim_robot.get_links_ang, None, None),
        (ezsim_robot.n_links, n_envs, ezsim_robot.get_links_acc, None, None),
        (ezsim_robot.n_links, n_envs, (), ezsim_robot.set_mass_shift, None),
        (ezsim_robot.n_links, n_envs, (3,), ezsim_robot.set_COM_shift, None),
        (ezsim_robot.n_links, n_envs, (), ezsim_robot.set_friction_ratio, None),
        (ezsim_robot.n_links, -1, ezsim_robot.get_links_inertial_mass, ezsim_robot.set_links_inertial_mass, None),
        (ezsim_robot.n_links, -1, ezsim_robot.get_links_invweight, ezsim_robot.set_links_invweight, None),
        (ezsim_robot.n_dofs, n_envs, ezsim_robot.get_dofs_control_force, None, None),
        (ezsim_robot.n_dofs, n_envs, ezsim_robot.get_dofs_force, None, None),
        (ezsim_robot.n_dofs, n_envs, ezsim_robot.get_dofs_velocity, ezsim_robot.set_dofs_velocity, None),
        (ezsim_robot.n_dofs, n_envs, ezsim_robot.get_dofs_position, ezsim_robot.set_dofs_position, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_force_range, ezsim_robot.set_dofs_force_range, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_limit, None, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_stiffness, None, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_invweight, None, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_armature, None, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_damping, None, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_kp, ezsim_robot.set_dofs_kp, None),
        (ezsim_robot.n_dofs, -1, ezsim_robot.get_dofs_kv, ezsim_robot.set_dofs_kv, None),
        (ezsim_robot.n_qs, n_envs, ezsim_robot.get_qpos, ezsim_robot.set_qpos, None),
        (-1, n_envs, ezsim_robot.get_mass_mat, None, None),
        (-1, n_envs, ezsim_robot.get_links_net_contact_force, None, None),
        (-1, n_envs, ezsim_robot.get_pos, ezsim_robot.set_pos, None),
        (-1, n_envs, ezsim_robot.get_quat, ezsim_robot.set_quat, None),
        (-1, -1, ezsim_robot.get_mass, ezsim_robot.set_mass, None),
        (-1, -1, ezsim_robot.get_AABB, None, None),
        # LINK
        (-1, -1, ezsim_link.get_mass, ezsim_link.set_mass, None),
    ):
        getter, spec = (getter_or_spec, None) if callable(getter_or_spec) else (None, getter_or_spec)
        # Check getter and setter without row or column masking
        if getter is not None:
            datas = getter()
            is_tuple = isinstance(datas, (tuple, list))
            if arg1_max > 0:
                assert_allclose(getter(range(arg1_max)), datas, tol=tol)
        else:
            batch_shape = []
            if arg2_max > 0:
                batch_shape.append(arg2_max)
            if arg1_max > 0:
                batch_shape.append(arg1_max)
            is_tuple = spec and isinstance(spec[0], (tuple, list))
            if is_tuple:
                datas = [torch.ones((*batch_shape, *shape)) for shape in spec]
            else:
                datas = torch.ones((*batch_shape, *spec))


        if field is not None:
            true = field.to_torch(device="cpu")
            true = true.movedim(true.ndim - getattr(field, "ndim", 0) - 1, 0)
            if is_tuple:
                true = torch.unbind(true, dim=-1)
                true = [val.reshape(data.shape) for data, val in zip(datas, true)]
            else:
                true = true.reshape(datas.shape)
            assert_allclose(datas, true, tol=tol)
        if setter is not None:
            if is_tuple:
                datas = [torch.as_tensor(val) for val in datas]
            else:
                datas = torch.as_tensor(datas)
            datas_tp = datas if is_tuple else (datas,)
            if getter is not None:
                # Randomly sample new data that are strictly positive and normalized,
                # as this may be required for some setters (mass, quaternion, ...).
                for val in datas_tp:
                    val[()] = torch.abs(torch.randn(val.shape, dtype=ezsim.tc_float, device=gs.device)) + gs.EPS
                    val /= torch.linalg.norm(val, dim=-1, keepdims=True)
            setter(*datas)
            if getter is not None:
                assert_allclose(getter(), datas, tol=tol)
         # Early return if neither rows or columns can be masked
        if not (arg1_max > 0 or arg2_max > 0):
            continue



        # Check getter and setter for all possible combinations of row and column masking
        for i in range(arg1_max) if arg1_max > 0 else (None,):
            for arg1 in get_all_supported_masks(i) if arg1_max > 0 else (None,):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    for arg2 in get_all_supported_masks(j) if arg2_max > 0 else (None,):
                        if arg1 is None and arg2 is not None:
                            unsafe = not must_cast(arg2)
                            if getter is not None:
                                data = getter(arg2, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, *spec))
                            if setter is not None:
                                setter(data, arg2, unsafe=unsafe)
                            if n_envs:
                                if is_tuple:
                                    data_ = [val[[j]] for val in datas]
                                else:
                                    data_ = datas[[j]]
                            else:
                                data_ = datas
                        elif arg1 is not None and arg2 is None:
                            unsafe = not must_cast(arg1)
                            if getter is not None:
                                data = getter(arg1, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1, unsafe=unsafe)
                                else:
                                    setter(data, arg1, unsafe=unsafe)
                            if is_tuple:
                                data_ = [val[[i]] for val in datas]
                            else:
                                data_ = datas[[i]]

                        else:
                            unsafe = not any(map(must_cast, (arg1, arg2)))
                            if getter is not None:
                                data = getter(arg1, arg2, unsafe=unsafe)
                            else:
                                if is_tuple:
                                    data = [torch.ones((1, 1, *shape)) for shape in spec]
                                else:
                                    data = torch.ones((1, 1, *spec))
                            if setter is not None:
                                setter(data, arg1, arg2, unsafe=unsafe)
                            if is_tuple:
                                data_ = [val[[j], :][:, [i]] for val in datas]
                            else:
                                data_ = datas[[j], :][:, [i]]
                        # FIXME: Not sure why tolerance must be increased for tests to pass
                        assert_allclose(data_, data, tol=(5.0 * tol))

    for dofs_idx in (*get_all_supported_masks(0), None):
        for envs_idx in (*(get_all_supported_masks(0) if n_envs > 0 else ()), None):
            unsafe = not any(map(must_cast, (dofs_idx, envs_idx)))
            dofs_pos = ezsim_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = ezsim_s.get_dofs_velocity(dofs_idx, envs_idx)
            ezsim_s.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            ezsim_s.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)
    # Must be tested independently because of non-trival return type
    ezsim_robot.get_contacts()


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_mesh_to_heightfield(tmp_path, show_viewer):
    horizontal_scale = 2.0
    path_terrain = os.path.join(get_assets_dir(), "meshes", "terrain_45.obj")

    hf_terrain, xs, ys = ezsim.utils.terrain.mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    ########################## create a scene ##########################
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            gravity=(2, 0, -2),
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    terrain_heightfield = scene.add_entity(
        morph=ezsim.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )
    ball = scene.add_entity(
        ezsim.morphs.Sphere(
            pos=(10, 15, 10),
            radius=1,
        ),
        vis_mode="collision",
    )
    scene.build()

    for i in range(1000):
        scene.step()

    # speed is around 0
    qvel = ball.get_dofs_velocity()
    assert_allclose(qvel, 0, atol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_get_cartesian_space_variables(show_viewer, tol):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=ezsim.options.RigidOptions(
            # by default, enable_mujoco_compatibility=False
            # the test will fail if enable_mujoco_compatibility=True
            enable_mujoco_compatibility=False,
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        ezsim.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    scene.build()

    for _ in range(2):
        for link in box.links:
            force = torch.tensor(np.array([0, 0, 0])).unsqueeze(0)
            acc = 50.0
            force[0, 0] = acc * link.inertial_mass
            pos = link.get_pos()
            vel = link.get_vel()

            dof_vel = link.solver.get_dofs_velocity()
            dof_pos = link.solver.get_qpos()

            assert_allclose(dof_vel[:3], vel, atol=tol)
            assert_allclose(dof_pos[:3], pos, atol=tol)

            link.solver.apply_links_external_force(force, (link.idx,), ref="link_com", local=False)

        scene.step()


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_geom_pos_quat(show_viewer, tol):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        ezsim.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 2.0),
        )
    )
    scene.build()

    for link in box.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            assert_allclose(geom.get_pos(), vgeom.get_pos(), atol=tol)
            assert_allclose(geom.get_quat(), vgeom.get_quat(), atol=tol)

@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_contype_conaffinity(show_viewer, tol):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        ezsim.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
        )
    )
    box1 = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 0.5),
            contype=3,
            conaffinity=3,
        )
    )
    box2 = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.0),
            contype=2,
            conaffinity=2,
        )
    )
    box3 = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 1.5),
            contype=1,
            conaffinity=1,
        )
    )
    scene.build()

    for _ in range(100):
        scene.step()

    assert_allclose(box2.get_pos(), box3.get_pos(), atol=1e-3)
    assert_allclose(box1.get_pos(), np.array([0.0, 0.0, 0.25]), atol=1e-3)
    assert_allclose(box2.get_pos(), np.array([0.0, 0.0, 0.75]), atol=1e-3)
    assert_allclose(box3.get_pos(), np.array([0.0, 0.0, 0.75]), atol=1e-3)
