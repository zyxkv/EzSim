import platform
import os
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Literal, Sequence

import cpuinfo
import numpy as np
import mujoco
import torch
from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError

import ezsim
import ezsim.utils.geom as gu
from ezsim.utils import mjcf as mju
from ezsim.utils.mesh import get_assets_dir
from ezsim.utils.misc import tensor_to_array


REPOSITY_URL = "Genesis-Embodied-AI/Genesis"
DEFAULT_BRANCH_NAME = "main"

# Get repository "root" path (actually test dir is good enough)
TEST_DIR = os.path.dirname(__file__)


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


@cache
def get_hardware_fingerprint(include_gpu=True):
    # CPU info
    cpu_info = cpuinfo.get_cpu_info()
    infos = [
        cpu_info.get("brand_raw", cpu_info.get("hardware_raw")),
        cpu_info.get("arch"),
    ]

    # GPU info
    if include_gpu and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        infos += [
            props.name,
            ".".join(map(str, (props.major, props.minor))),
            props.total_memory,
            props.multi_processor_count,  # Number of "streaming multiprocessors"
        ]

    return "-".join(map(str, filter(None, infos)))


@cache
def get_platform_fingerprint():
    # OS distribution info
    system = platform.system()
    dist_name = None
    if system == "Linux":
        try:
            dist_info = platform.freedesktop_os_release()
            dist_name = dist_info["ID"]
            dist_ver = dist_info["VERSION_ID"]
        except FileNotFoundError:
            pass
    elif system == "Darwin":
        dist_name = "MacOS"
        dist_ver, *_ = platform.mac_ver()
    if dist_name is None:
        dist_name = system
        dist_ver, *_ = platform.release().split(".", 1)  # Only extract major version.

    infos = [
        dist_name,
        dist_ver,  # Only extract major version.
    ]

    # Python info
    py_major, py_minor, py_patchlevel = platform.python_version_tuple()
    infos += [
        ".".join((py_major, py_minor)),  # Ignore patch-level version
    ]

    return "-".join(map(str, filter(None, infos)))


@cache
def get_git_commit_timestamp(ref="HEAD"):
    try:
        contrib_date = subprocess.check_output(
            ["git", "show", "-s", "--quiet", "--format=%ci", ref], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        # Commit not found, either because it does not exist or becaused fo shallow git clone
        return float("nan")

    try:
        date = datetime.fromisoformat(contrib_date)
    except ValueError:
        date = datetime.strptime(contrib_date, "%Y-%m-%d %H:%M:%S %z")
    timestamp = date.timestamp()

    return timestamp


@cache
def get_git_commit_info(ref="HEAD"):
    # Fetch current commit revision
    try:
        revision = subprocess.check_output(["git", "rev-parse", ref], cwd=TEST_DIR, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        revision = f"{uuid.uuid4().hex}@UNKNOWN"
        timestamp = float("nan")
        return revision, timestamp

    # Fetch all remote branches containing the current commit
    try:
        branches = subprocess.check_output(
            ["git", "branch", "--remote", "--contains", ref], cwd=TEST_DIR, encoding="utf-8"
        ).splitlines()
    except subprocess.CalledProcessError:
        # Raise error if not found neither locally nor remotely
        branches = ()

    # Check if the current commit is contained by main branch
    remote_handle = "UNKNOWN"
    for branch in branches:
        try:
            remote_name, branch_name = branch.strip().split("/", 1)
        except ValueError:
            continue
        if branch_name != DEFAULT_BRANCH_NAME:
            continue
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", remote_name], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
        if remote_url.startswith("https://github.com/"):
            remote_handle = remote_url[19:-4]
        elif remote_url.startswith("git@github.com:"):
            remote_handle = remote_url[15:-4]
        if remote_handle == REPOSITY_URL:
            is_commit_on_default_branch = True
            break
    else:
        is_commit_on_default_branch = False

    # Return the contribution date as timestamp if and only if the HEAD commit is contained on main branch
    if is_commit_on_default_branch:
        timestamp = get_git_commit_timestamp(ref)
        return revision, timestamp

    revision = f"{revision}@{remote_handle}"
    timestamp = float("nan")
    return revision, timestamp


def get_hf_assets(pattern, num_retry: int = 4, retry_delay: float = 30.0, check: bool = True):
    assert num_retry >= 1

    for _ in range(num_retry):
        num_trials = 0
        try:
            # Try downloading the assets
            asset_path = snapshot_download(
                repo_type="dataset",
                repo_id="Genesis-Intelligence/assets",
                allow_patterns=pattern,
                max_workers=1,
            )

            # Make sure that download was successful
            has_files = False
            for path in Path(asset_path).rglob(pattern):
                if not path.is_file():
                    continue
                has_files = True

                if path.stat().st_size == 0:
                    raise HTTPError(f"File '{path}' is empty.")

                if path.suffix.lower() in (".xml", ".urdf"):
                    try:
                        ET.parse(path)
                    except ET.ParseError as e:
                        raise HTTPError(f"Impossible to parse XML file.") from e
            if not has_files:
                raise HTTPError("No file downloaded.")
        except HTTPError:
            num_trials += 1
            if num_trials == num_retry:
                raise
            print(f"Failed to download assets from HuggingFace dataset. Trying again in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            break

    return asset_path


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=""):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    args = [actual, desired]
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            arg = tensor_to_array(arg)
        elif isinstance(arg, (tuple, list)):
            arg = [tensor_to_array(val) for val in arg]
        args[i] = np.asanyarray(arg)

    if all(e.size == 0 for e in args):
        return

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)

def assert_array_equal(actual, desired, *, err_msg=""):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)


def init_simulators(ezsim_sim, mj_sim=None, qpos=None, qvel=None):
    if mj_sim is not None:
        _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(ezsim_sim, mj_sim)

    (ezsim_robot,) = ezsim_sim.entities

    ezsim_sim.scene.reset()
    if qpos is not None:
        ezsim_robot.set_qpos(qpos)
    if qvel is not None:
        ezsim_robot.set_dofs_velocity(qvel)
    # TODO: This should be moved in `set_state`, `set_qpos`, `set_dofs_position`, `set_dofs_velocity`
    ezsim_sim.rigid_solver.dofs_state.qf_constraint.fill(0.0)
    ezsim_sim.rigid_solver._func_forward_dynamics()
    ezsim_sim.rigid_solver._func_constraint_force()
    ezsim_sim.rigid_solver._func_update_acc()

    if ezsim_sim.scene.visualizer:
        ezsim_sim.scene.visualizer.update()

    if mj_sim is not None:
        mujoco.mj_resetData(mj_sim.model, mj_sim.data)
        mj_sim.data.qpos[mj_qs_idx] = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = ezsim_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mujoco.mj_forward(mj_sim.model, mj_sim.data)


def _ezsim_search_by_joints_name(
    scene,
    joints_name: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(joints_name, str):
        joints_name = [joints_name]

    for entity in scene.entities:
        try:
            ezsim_joints_idx = dict()
            ezsim_joints_qs_idx = dict()
            ezsim_joints_dofs_idx = dict()
            valid_joints_name = []
            for joint in entity.joints:
                valid_joints_name.append(joint.name)
                if joint.name in joints_name:
                    if to == "entity":
                        ezsim_joints_idx[joint.name] = joint
                        ezsim_joints_qs_idx[joint.name] = joint
                        ezsim_joints_dofs_idx[joint.name] = joint
                    elif to == "index":
                        ezsim_joints_idx[joint.name] = joint.idx_local if is_local else joint.idx
                        ezsim_joints_qs_idx[joint.name] = joint.qs_idx_local if is_local else joint.qs_idx
                        ezsim_joints_dofs_idx[joint.name] = joint.dofs_idx_local if is_local else joint.dofs_idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_joints_name = set(joints_name) - ezsim_joints_idx.keys()
            if len(missing_joints_name) > 0:
                raise ValueError(
                    f"Cannot find joints `{missing_joints_name}`. Valid joints names are {valid_joints_name}"
                )

            if flatten:
                return (
                    list(ezsim_joints_idx.values()),
                    list(chain.from_iterable(ezsim_joints_qs_idx.values())),
                    list(chain.from_iterable(ezsim_joints_dofs_idx.values())),
                )
            return (ezsim_joints_idx, ezsim_joints_qs_idx, ezsim_joints_dofs_idx)
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find joint indices for {joints_name}")


def _ezsim_search_by_links_name(
    scene,
    links_name: str | Sequence[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(links_name, str):
        links_name = (links_name,)

    for entity in scene.entities:
        try:
            ezsim_links_idx = dict()
            valid_links_name = []
            for link in entity.links:
                valid_links_name.append(link.name)
                if link.name in links_name:
                    if to == "entity":
                        ezsim_links_idx[link.name] = link
                    elif to == "index":
                        ezsim_links_idx[link.name] = link.idx_local if is_local else link.idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_links_name = set(links_name) - ezsim_links_idx.keys()
            if missing_links_name:
                raise ValueError(f"Cannot find links `{missing_links_name}`. Valid link names are {valid_links_name}")

            if flatten:
                return list(ezsim_links_idx.values())
            return ezsim_links_idx
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find link indices for {links_name}")


def _get_model_mappings(
    ezsim_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
):
    if joints_name is None:
        joints_name = [
            joint.name for entity in ezsim_sim.entities for joint in entity.joints if joint.type != ezsim.JOINT_TYPE.FIXED
        ]
    if bodies_name is None:
        bodies_name = [
            body.name
            for entity in ezsim_sim.entities
            for body in entity.links
            if not (body.is_fixed and body.parent_idx < 0)
        ]

    motors_name: list[str] = []
    mj_joints_idx: list[int] = []
    mj_qs_idx: list[int] = []
    mj_dofs_idx: list[int] = []
    mj_geoms_idx: list[int] = []
    mj_motors_idx: list[int] = []
    for joint_name in joints_name:
        if joint_name:
            mj_joint = mj_sim.model.joint(joint_name)
        else:
            # Must rely on exhaustive search if the joint has empty name
            for j in range(mj_sim.model.njoint):
                mj_joint = mj_sim.model.joint(j)
                if mj_joint.name == "":
                    break
            else:
                raise ValueError(f"Invalid joint name '{joint_name}'.")
        mj_joints_idx.append(mj_joint.id)
        mj_type = mj_sim.model.jnt_type[mj_joint.id]
        if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            n_dofs, n_qs = 3, 4
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            n_dofs, n_qs = 6, 7
        else:
            raise ValueError(f"Invalid joint type '{mj_type}'.")
        mj_dof_start_j = mj_sim.model.jnt_dofadr[mj_joint.id]
        mj_dofs_idx += range(mj_dof_start_j, mj_dof_start_j + n_dofs)

        mj_q_start_j = mj_sim.model.jnt_qposadr[mj_joint.id]
        mj_qs_idx += range(mj_q_start_j, mj_q_start_j + n_qs)
        if (mj_joint.id == mj_sim.model.actuator_trnid[:, 0]).any():
            motors_name.append(joint_name)
            (motors_idx,) = np.nonzero(mj_joint.id == mj_sim.model.actuator_trnid[:, 0])
            # FIXME: only supporting 1DoF per actuator
            mj_motors_idx.append(motors_idx[0])

    mj_bodies_idx, mj_geoms_idx = [], []
    for body_name in bodies_name:
        mj_body = mj_sim.model.body(body_name)
        mj_bodies_idx.append(mj_body.id)
        for mj_geom_idx in range(mj_body.geomadr[0], mj_body.geomadr[0] + mj_body.geomnum[0]):
            mj_geom = mj_sim.model.geom(mj_geom_idx)
            if mj_geom.contype or mj_geom.conaffinity:
                mj_geoms_idx.append(mj_geom.id)

    (ezsim_joints_idx, ezsim_q_idx, ezsim_dofs_idx) = _ezsim_search_by_joints_name(ezsim_sim.scene, joints_name)
    (_, _, ezsim_motors_idx) = _ezsim_search_by_joints_name(ezsim_sim.scene, motors_name)

    ezsim_bodies_idx = _ezsim_search_by_links_name(ezsim_sim.scene, bodies_name)
    ezsim_geoms_idx: list[int] = []
    for ezsim_body_idx in ezsim_bodies_idx:
        link = ezsim_sim.rigid_solver.links[ezsim_body_idx]
        ezsim_geoms_idx += range(link.geom_start, link.geom_end)

    ezsim_maps = (ezsim_bodies_idx, ezsim_joints_idx, ezsim_q_idx, ezsim_dofs_idx, ezsim_geoms_idx, ezsim_motors_idx)
    mj_maps = (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx)
    return ezsim_maps, mj_maps


def build_mujoco_sim(
    xml_path, ezsim_solver, ezsim_integrator, merge_fixed_links, multi_contact, adjacent_collision, dof_damping, native_ccd
):
    if ezsim_solver == ezsim.constraint_solver.CG:
        mj_solver = mujoco.mjtSolver.mjSOL_CG
    elif ezsim_solver == ezsim.constraint_solver.Newton:
        mj_solver = mujoco.mjtSolver.mjSOL_NEWTON
    else:
        raise ValueError(f"Solver '{ezsim_solver}' not supported")
    if ezsim_integrator == ezsim.integrator.Euler:
        mj_integrator = mujoco.mjtIntegrator.mjINT_EULER
    elif ezsim_integrator == ezsim.integrator.implicitfast:
        mj_integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        raise ValueError(f"Integrator '{ezsim_integrator}' not supported")

    xml_path = os.path.join(get_assets_dir(), xml_path)
    model = mju.build_model(
        xml_path, discard_visual=True, default_armature=None, merge_fixed_links=merge_fixed_links, links_to_keep=()
    )

    model.opt.solver = mj_solver
    model.opt.integrator = mj_integrator
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    if native_ccd:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    else:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    if multi_contact:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    else:
        model.opt.enableflags &= ~np.uint32(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    if adjacent_collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    else:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    data = mujoco.MjData(model)

    return MjSim(model, data)


def build_ezsim_sim(
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
):
    scene = ezsim.Scene(
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=ezsim.options.SimOptions(
            dt=mj_sim.model.opt.timestep,
            substeps=1,
            gravity=mj_sim.model.opt.gravity.tolist(),
        ),
        rigid_options=ezsim.options.RigidOptions(
            integrator=ezsim_integrator,
            constraint_solver=ezsim_solver,
            enable_mujoco_compatibility=mujoco_compatibility,
            box_box_detection=True,
            enable_self_collision=True,
            enable_adjacent_collision=adjacent_collision,
            enable_multi_contact=multi_contact,
            iterations=mj_sim.model.opt.iterations,
            tolerance=mj_sim.model.opt.tolerance,
            ls_iterations=mj_sim.model.opt.ls_iterations,
            ls_tolerance=mj_sim.model.opt.ls_tolerance,
            use_gjk_collision=gjk_collision,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    morph_kwargs = dict(
        file=xml_path,
        convexify=True,
        decompose_robot_error_threshold=float("inf"),
        default_armature=None,
    )
    if xml_path.endswith(".xml"):
        morph = ezsim.morphs.MJCF(**morph_kwargs)
    else:
        morph = ezsim.morphs.URDF(
            fixed=True,
            merge_fixed_links=merge_fixed_links,
            links_to_keep=(),
            **morph_kwargs,
        )
    ezsim_robot = scene.add_entity(
        morph,
        visualize_contact=True,
    )

    # Force matching Mujoco safety factor for constraint time constant.
    # Note that this time constant affects the penetration depth at rest.
    ezsim_sim = scene.sim
    ezsim_sim.rigid_solver._sol_default_timeconst = None
    ezsim_sim.rigid_solver._sol_min_timeconst = 2.0 * ezsim_sim._substep_dt

    # Force recomputation of invweights to make sure it works fine
    for link in scene.rigid_solver.links:
        link.invweight[:] = -1
    for joint in scene.rigid_solver.joints:
        joint.dofs_invweight[:] = -1

    scene.build()

    return ezsim_sim


def check_mujoco_model_consistency(
    ezsim_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    tol: float,
):
    # Delay import to enable run benchmarks for old Genesis versions that do not have this method
    from ezsim.engine.solvers.rigid.rigid_solver_decomp import _sanitize_sol_params

    # Get mapping between Mujoco and Genesis
    ezsim_maps, mj_maps = _get_model_mappings(ezsim_sim, mj_sim, joints_name, bodies_name)
    (ezsim_bodies_idx, ezsim_joints_idx, ezsim_q_idx, ezsim_dofs_idx, ezsim_geoms_idx, ezsim_motors_idx) = ezsim_maps
    (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_geoms_idx, mj_motors_idx) = mj_maps

    # solver
    ezsim_gravity = ezsim_sim.rigid_solver.scene.gravity
    mj_gravity = mj_sim.model.opt.gravity
    assert_allclose(ezsim_gravity, mj_gravity, tol=tol)
    assert mj_sim.model.opt.timestep == ezsim_sim.rigid_solver.substep_dt
    assert mj_sim.model.opt.tolerance == ezsim_sim.rigid_solver._options.tolerance
    assert mj_sim.model.opt.iterations == ezsim_sim.rigid_solver._options.iterations
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    assert not (mj_sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV)

    mj_adj_collision = bool(mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    ezsim_adj_collision = ezsim_sim.rigid_solver._options.enable_adjacent_collision
    assert ezsim_adj_collision == mj_adj_collision

    ezsim_use_gjk_collision = ezsim_sim.rigid_solver._options.use_gjk_collision
    mj_use_gjk_collision = not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    assert ezsim_use_gjk_collision == mj_use_gjk_collision

    mj_solver = mujoco.mjtSolver(mj_sim.model.opt.solver)
    if mj_solver.name == "mjSOL_PGS":
        assert False
    elif mj_solver.name == "mjSOL_CG":
        assert ezsim_sim.rigid_solver._options.constraint_solver == ezsim.constraint_solver.CG
    elif mj_solver.name == "mjSOL_NEWTON":
        assert ezsim_sim.rigid_solver._options.constraint_solver == ezsim.constraint_solver.Newton
    else:
        assert False

    mj_integrator = mujoco.mjtIntegrator(mj_sim.model.opt.integrator)
    if mj_integrator.name == "mjINT_EULER":
        assert ezsim_sim.rigid_solver._options.integrator == ezsim.integrator.Euler
    elif mj_integrator.name == "mjINT_IMPLICIT":
        assert False
    elif mj_integrator.name == "mjINT_IMPLICITFAST":
        assert ezsim_sim.rigid_solver._options.integrator == ezsim.integrator.implicitfast
    else:
        assert False

    mj_cone = mujoco.mjtCone(mj_sim.model.opt.cone)
    if mj_cone.name == "mjCONE_ELLIPTIC":
        assert False
    elif mj_cone.name == "mjCONE_PYRAMIDAL":
        assert True
    else:
        assert False

    ezsim_roots_name = sorted(
        ezsim_sim.rigid_solver.links[i].name
        for i in set(ezsim_sim.rigid_solver.links_info.root_idx.to_numpy()[ezsim_bodies_idx])
    )
    mj_roots_name = sorted(mj_sim.model.body(i).name for i in set(mj_sim.model.body_rootid[mj_bodies_idx]))
    assert ezsim_roots_name == mj_roots_name

    # body
    for ezsim_i, mj_i in zip(ezsim_bodies_idx, mj_bodies_idx):
        ezsim_invweight_i = ezsim_sim.rigid_solver.links_info.invweight.to_numpy()[ezsim_i]
        mj_invweight_i = mj_sim.model.body(mj_i).invweight0
        assert_allclose(ezsim_invweight_i, mj_invweight_i, tol=tol)
        ezsim_inertia_i = ezsim_sim.rigid_solver.links_info.inertial_i.to_numpy()[ezsim_i, [0, 1, 2], [0, 1, 2]]
        mj_inertia_i = mj_sim.model.body(mj_i).inertia
        assert_allclose(ezsim_inertia_i, mj_inertia_i, tol=tol)
        ezsim_ipos_i = ezsim_sim.rigid_solver.links_info.inertial_pos.to_numpy()[ezsim_i]
        mj_ipos_i = mj_sim.model.body(mj_i).ipos
        assert_allclose(ezsim_ipos_i, mj_ipos_i, tol=tol)
        ezsim_iquat_i = ezsim_sim.rigid_solver.links_info.inertial_quat.to_numpy()[ezsim_i]
        mj_iquat_i = mj_sim.model.body(mj_i).iquat
        assert_allclose(ezsim_iquat_i, mj_iquat_i, tol=tol)
        ezsim_pos_i = ezsim_sim.rigid_solver.links_info.pos.to_numpy()[ezsim_i]
        mj_pos_i = mj_sim.model.body(mj_i).pos
        assert_allclose(ezsim_pos_i, mj_pos_i, tol=tol)
        ezsim_quat_i = ezsim_sim.rigid_solver.links_info.quat.to_numpy()[ezsim_i]
        mj_quat_i = mj_sim.model.body(mj_i).quat
        assert_allclose(ezsim_quat_i, mj_quat_i, tol=tol)
        ezsim_mass_i = ezsim_sim.rigid_solver.links_info.inertial_mass.to_numpy()[ezsim_i]
        mj_mass_i = mj_sim.model.body(mj_i).mass
        assert_allclose(ezsim_mass_i, mj_mass_i, tol=tol)

    # dof / joints
    ezsim_dof_damping = ezsim_sim.rigid_solver.dofs_info.damping.to_numpy()
    mj_dof_damping = mj_sim.model.dof_damping
    assert_allclose(ezsim_dof_damping[ezsim_dofs_idx], mj_dof_damping[mj_dofs_idx], tol=tol)

    ezsim_dof_armature = ezsim_sim.rigid_solver.dofs_info.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    assert_allclose(ezsim_dof_armature[ezsim_dofs_idx], mj_dof_armature[mj_dofs_idx], tol=tol)

    # FIXME: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    ezsim_dof_stiffness = ezsim_sim.rigid_solver.dofs_info.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    # assert_allclose(ezsim_dof_stiffness[ezsim_dofs_idx], mj_dof_stiffness[mj_joints_idx], tol=tol)

    ezsim_dof_invweight0 = ezsim_sim.rigid_solver.dofs_info.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    assert_allclose(ezsim_dof_invweight0[ezsim_dofs_idx], mj_dof_invweight0[mj_dofs_idx], tol=tol)

    # TODO: Genesis does not support frictionloss contraint at dof level for now
    ezsim_joint_solparams = np.array([joint.sol_params.cpu() for entity in ezsim_sim.entities for joint in entity.joints])
    mj_joint_solparams = np.concatenate((mj_sim.model.jnt_solref, mj_sim.model.jnt_solimp), axis=-1)
    _sanitize_sol_params(
        mj_joint_solparams, ezsim_sim.rigid_solver._sol_min_timeconst, ezsim_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(ezsim_joint_solparams[ezsim_joints_idx], mj_joint_solparams[mj_joints_idx], tol=tol)
    ezsim_geom_solparams = np.array([geom.sol_params.cpu() for entity in ezsim_sim.entities for geom in entity.geoms])
    mj_geom_solparams = np.concatenate((mj_sim.model.geom_solref, mj_sim.model.geom_solimp), axis=-1)
    _sanitize_sol_params(
        mj_geom_solparams, ezsim_sim.rigid_solver._sol_min_timeconst, ezsim_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(ezsim_geom_solparams[ezsim_geoms_idx], mj_geom_solparams[mj_geoms_idx], tol=tol)
    # FIXME: Masking geometries and equality constraints is not supported for now
    ezsim_eq_solparams = np.array(
        [equality.sol_params.cpu() for entity in ezsim_sim.entities for equality in entity.equalities]
    ).reshape((-1, 7))
    mj_eq_solparams = np.concatenate((mj_sim.model.eq_solref, mj_sim.model.eq_solimp), axis=-1)
    _sanitize_sol_params(
        mj_eq_solparams, ezsim_sim.rigid_solver._sol_min_timeconst, ezsim_sim.rigid_solver._sol_default_timeconst
    )
    assert_allclose(ezsim_eq_solparams, mj_eq_solparams, tol=tol)

    assert_allclose(mj_sim.model.jnt_margin, 0, tol=tol)
    ezsim_joint_range = np.stack(
        [
            ezsim_sim.rigid_solver.dofs_info.limit[ezsim_sim.rigid_solver.joints_info.dof_start[i]].to_numpy()
            for i in ezsim_joints_idx
        ],
        axis=0,
    )
    mj_joint_range = mj_sim.model.jnt_range
    mj_joint_range[mj_sim.model.jnt_limited == 0, 0] = float("-inf")
    mj_joint_range[mj_sim.model.jnt_limited == 0, 1] = float("+inf")
    assert_allclose(ezsim_joint_range, mj_joint_range[mj_joints_idx], tol=tol)

    # actuator (position control)
    for v in mj_sim.model.actuator_dyntype:
        assert v == mujoco.mjtDyn.mjDYN_NONE
    for v in mj_sim.model.actuator_biastype:
        assert v in (mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtBias.mjBIAS_NONE)

    # NOTE: not considering gear
    ezsim_kp = ezsim_sim.rigid_solver.dofs_info.kp.to_numpy()
    ezsim_kv = ezsim_sim.rigid_solver.dofs_info.kv.to_numpy()
    mj_kp = -mj_sim.model.actuator_biasprm[:, 1]
    mj_kv = -mj_sim.model.actuator_biasprm[:, 2]
    assert_allclose(ezsim_kp[ezsim_motors_idx], mj_kp[mj_motors_idx], tol=tol)
    assert_allclose(ezsim_kv[ezsim_motors_idx], mj_kv[mj_motors_idx], tol=tol)


def check_mujoco_data_consistency(
    ezsim_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    bodies_name: list[str] | None = None,
    *,
    qvel_prev: np.ndarray | None = None,
    tol: float,
):
    # Get mapping between Mujoco and Genesis
    ezsim_maps, mj_maps = _get_model_mappings(ezsim_sim, mj_sim, joints_name, bodies_name)
    (ezsim_bodies_idx, _, ezsim_q_idx, ezsim_dofs_idx, _, _) = ezsim_maps
    (mj_bodies_idx, _, mj_qs_idx, mj_dofs_idx, _, _) = mj_maps

    # crb
    ezsim_crb_inertial = ezsim_sim.rigid_solver.links_state.crb_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[:, :6]  # upper-triangular part
    assert_allclose(ezsim_crb_inertial[ezsim_bodies_idx], mj_crb_inertial[mj_bodies_idx], tol=tol)
    ezsim_crb_pos = ezsim_sim.rigid_solver.links_state.crb_pos.to_numpy()[:, 0]
    mj_crb_pos = mj_sim.data.crb[:, 6:9]
    assert_allclose(ezsim_crb_pos[ezsim_bodies_idx], mj_crb_pos[mj_bodies_idx], tol=tol)
    ezsim_crb_mass = ezsim_sim.rigid_solver.links_state.crb_mass.to_numpy()[:, 0]
    mj_crb_mass = mj_sim.data.crb[:, 9]
    assert_allclose(ezsim_crb_mass[ezsim_bodies_idx], mj_crb_mass[mj_bodies_idx], tol=tol)

    ezsim_mass_mat = ezsim_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat = np.zeros((mj_sim.model.nv, mj_sim.model.nv))
    mujoco.mj_fullM(mj_sim.model, mj_mass_mat, mj_sim.data.qM)
    assert_allclose(ezsim_mass_mat[ezsim_dofs_idx][:, ezsim_dofs_idx], mj_mass_mat[mj_dofs_idx][:, mj_dofs_idx], tol=tol)

    ezsim_meaninertia = ezsim_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    assert_allclose(ezsim_meaninertia, mj_meaninertia, tol=tol)

    # Pre-constraint so-called bias forces in configuration space
    ezsim_qfrc_bias = ezsim_sim.rigid_solver.dofs_state.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    assert_allclose(ezsim_qfrc_bias, mj_qfrc_bias[mj_dofs_idx], tol=tol)
    ezsim_qfrc_passive = ezsim_sim.rigid_solver.dofs_state.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    assert_allclose(ezsim_qfrc_passive, mj_qfrc_passive[mj_dofs_idx], tol=tol)
    ezsim_qfrc_actuator = ezsim_sim.rigid_solver.dofs_state.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    assert_allclose(ezsim_qfrc_actuator, mj_qfrc_actuator[mj_dofs_idx], tol=tol)

    ezsim_n_contacts = ezsim_sim.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    assert ezsim_n_contacts == mj_n_contacts
    ezsim_n_constraints = ezsim_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert ezsim_n_constraints == mj_n_constraints

    if ezsim_n_constraints:
        ezsim_contact_pos = ezsim_sim.rigid_solver.collider._collider_state.contact_data.pos.to_numpy()[:ezsim_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        # Sort based on the axis with the largest variation
        max_var_axis = 0
        if ezsim_n_contacts > 1:
            max_var = -1
            for axis in range(3):
                sorted_contact_pos = np.sort(mj_contact_pos[:, axis])
                var = np.min(sorted_contact_pos[1:] - sorted_contact_pos[:-1])
                if var > max_var:
                    max_var_axis = axis
                    max_var = var
        ezsim_sidx = np.argsort(ezsim_contact_pos[:, max_var_axis])
        mj_sidx = np.argsort(mj_contact_pos[:, max_var_axis])
        assert_allclose(ezsim_contact_pos[ezsim_sidx], mj_contact_pos[mj_sidx], tol=tol)
        ezsim_contact_normal = ezsim_sim.rigid_solver.collider._collider_state.contact_data.normal.to_numpy()[
            :ezsim_n_contacts, 0
        ]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        assert_allclose(ezsim_contact_normal[ezsim_sidx], mj_contact_normal[mj_sidx], tol=tol)
        ezsim_penetration = ezsim_sim.rigid_solver.collider._collider_state.contact_data.penetration.to_numpy()[
            :ezsim_n_contacts, 0
        ]
        mj_penetration = -mj_sim.data.contact.dist
        assert_allclose(ezsim_penetration[ezsim_sidx], mj_penetration[mj_sidx], tol=tol)

        # FIXME: It is not always possible to reshape Mujoco jacobian because joint bound constraints are computed in
        # "sparse" dof space, unlike contact constraints.
        error = None
        ezsim_jac = ezsim_sim.rigid_solver.constraint_solver.jac.to_numpy()[:ezsim_n_constraints, :, 0]
        mj_jac = mj_sim.data.efc_J.reshape([mj_n_constraints, -1])
        ezsim_efc_D = ezsim_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:ezsim_n_constraints, 0]
        mj_efc_D = mj_sim.data.efc_D
        ezsim_efc_aref = ezsim_sim.rigid_solver.constraint_solver.aref.to_numpy()[:ezsim_n_constraints, 0]
        mj_efc_aref = mj_sim.data.efc_aref
        for ezsim_sidx, mj_sidx in (
            (np.argsort(ezsim_jac.sum(axis=1)), np.argsort(mj_jac.sum(axis=1))),
            (np.argsort(ezsim_efc_aref), np.argsort(mj_efc_aref)),
        ):
            try:
                ezsim_jac_nz_mask = (np.abs(ezsim_jac[ezsim_sidx]) > 0.0).all(axis=0)
                ezsim_jac_nz = ezsim_jac[ezsim_sidx][:, np.array(ezsim_dofs_idx)[ezsim_jac_nz_mask[ezsim_dofs_idx]]]
                mj_jac_nz_mask = np.zeros_like(ezsim_jac_nz_mask, dtype=np.bool_)
                mj_jac_nz_mask[mj_dofs_idx] = ezsim_jac_nz_mask[ezsim_dofs_idx]
                if mj_jac.shape[-1] == len(mj_dofs_idx):
                    mj_jac_nz = mj_jac[mj_sidx][:, np.array(mj_dofs_idx)[mj_jac_nz_mask[mj_dofs_idx]]]
                else:
                    mj_jac_nz = mj_jac[mj_sidx]

                assert_allclose(ezsim_jac_nz, mj_jac_nz, tol=tol)
                assert_allclose(ezsim_efc_D[ezsim_sidx], mj_efc_D[mj_sidx], tol=tol)
                assert_allclose(ezsim_efc_aref[ezsim_sidx], mj_efc_aref[mj_sidx], tol=tol)
                break
            except AssertionError as e:
                error = e
        else:
            assert error is not None
            raise error

        ezsim_efc_force = ezsim_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:ezsim_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        assert_allclose(ezsim_efc_force[ezsim_sidx], mj_efc_force[mj_sidx], tol=tol)

    if ezsim_n_constraints:
        mj_iter = mj_sim.data.solver_niter[0] - 1
        if ezsim_n_constraints and mj_iter >= 0:
            ezsim_scale = 1.0 / (ezsim_meaninertia * max(1, ezsim_sim.rigid_solver.n_dofs))
            ezsim_gradient = ezsim_scale * np.linalg.norm(
                ezsim_sim.rigid_solver.constraint_solver.grad.to_numpy()[: ezsim_sim.rigid_solver.n_dofs, 0]
            )
            mj_gradient = mj_sim.data.solver.gradient[mj_iter]
            assert_allclose(ezsim_gradient, mj_gradient, tol=tol)
            ezsim_improvement = ezsim_scale * (
                ezsim_sim.rigid_solver.constraint_solver.prev_cost[0] - ezsim_sim.rigid_solver.constraint_solver.cost[0]
            )
            mj_improvement = mj_sim.data.solver.improvement[mj_iter]
            # FIXME: This is too challenging to match because of compounding of errors
            # assert_allclose(ezsim_improvement, mj_improvement, tol=tol)

        if qvel_prev is not None:
            ezsim_efc_vel = ezsim_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            assert_allclose(ezsim_efc_vel[ezsim_sidx], mj_efc_vel[mj_sidx], tol=tol)

    ezsim_qfrc_constraint = ezsim_sim.rigid_solver.dofs_state.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    assert_allclose(ezsim_qfrc_constraint[ezsim_dofs_idx], mj_qfrc_constraint[mj_dofs_idx], tol=tol)

    ezsim_qfrc_all = ezsim_sim.rigid_solver.dofs_state.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    assert_allclose(ezsim_qfrc_all[ezsim_dofs_idx], mj_qfrc_all[mj_dofs_idx], tol=tol)

    ezsim_qfrc_smooth = ezsim_sim.rigid_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    assert_allclose(ezsim_qfrc_smooth[ezsim_dofs_idx], mj_qfrc_smooth[mj_dofs_idx], tol=tol)

    ezsim_qacc_smooth = ezsim_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    assert_allclose(ezsim_qacc_smooth[ezsim_dofs_idx], mj_qacc_smooth[mj_dofs_idx], tol=tol)

    # Acceleration pre- VS post-implicit damping
    # ezsim_qacc_post = ezsim_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]
    if ezsim_n_constraints:
        ezsim_qacc_pre = ezsim_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    else:
        ezsim_qacc_pre = ezsim_qacc_smooth
    mj_qacc_pre = mj_sim.data.qacc
    assert_allclose(ezsim_qacc_pre[ezsim_dofs_idx], mj_qacc_pre[mj_dofs_idx], tol=tol)

    ezsim_qvel = ezsim_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    assert_allclose(ezsim_qvel[ezsim_dofs_idx], mj_qvel[mj_dofs_idx], tol=tol)
    ezsim_qpos = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    assert_allclose(ezsim_qpos[ezsim_q_idx], mj_qpos[mj_qs_idx], tol=tol)

    # ------------------------------------------------------------------------

    ezsim_com = ezsim_sim.rigid_solver.links_state.COM.to_numpy()[:, 0]
    ezsim_root_idx = np.unique(ezsim_sim.rigid_solver.links_info.root_idx.to_numpy()[ezsim_bodies_idx])
    mj_com = mj_sim.data.subtree_com
    mj_root_idx = np.unique(mj_sim.model.body_rootid[mj_bodies_idx])
    assert_allclose(ezsim_com[ezsim_root_idx], mj_com[mj_root_idx], tol=tol)

    ezsim_xipos = ezsim_sim.rigid_solver.links_state.i_pos.to_numpy()[:, 0]
    mj_xipos = mj_sim.data.xipos - mj_sim.data.subtree_com[mj_sim.model.body_rootid]
    assert_allclose(ezsim_xipos[ezsim_bodies_idx], mj_xipos[mj_bodies_idx], tol=tol)

    ezsim_xpos = ezsim_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    mj_xpos = mj_sim.data.xpos
    assert_allclose(ezsim_xpos[ezsim_bodies_idx], mj_xpos[mj_bodies_idx], tol=tol)

    ezsim_xquat = ezsim_sim.rigid_solver.links_state.quat.to_numpy()[:, 0]
    ezsim_xmat = gu.quat_to_R(ezsim_xquat).reshape([-1, 9])
    mj_xmat = mj_sim.data.xmat
    assert_allclose(ezsim_xmat[ezsim_bodies_idx], mj_xmat[mj_bodies_idx], tol=tol)

    ezsim_cd_vel = ezsim_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    mj_cd_vel = mj_sim.data.cvel[:, 3:]
    assert_allclose(ezsim_cd_vel[ezsim_bodies_idx], mj_cd_vel[mj_bodies_idx], tol=tol)
    ezsim_cd_ang = ezsim_sim.rigid_solver.links_state.cd_ang.to_numpy()[:, 0]
    mj_cd_ang = mj_sim.data.cvel[:, :3]
    assert_allclose(ezsim_cd_ang[ezsim_bodies_idx], mj_cd_ang[mj_bodies_idx], tol=tol)

    ezsim_cdof_vel = ezsim_sim.rigid_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    assert_allclose(ezsim_cdof_vel[ezsim_dofs_idx], mj_cdof_vel[mj_dofs_idx], tol=tol)
    ezsim_cdof_ang = ezsim_sim.rigid_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    assert_allclose(ezsim_cdof_ang[ezsim_dofs_idx], mj_cdof_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_ang = mj_sim.data.cdof_dot[:, :3]
    ezsim_cdof_dot_ang = ezsim_sim.rigid_solver.dofs_state.cdofd_ang.to_numpy()[:, 0]
    assert_allclose(ezsim_cdof_dot_ang[ezsim_dofs_idx], mj_cdof_dot_ang[mj_dofs_idx], tol=tol)

    mj_cdof_dot_vel = mj_sim.data.cdof_dot[:, 3:]
    ezsim_cdof_dot_vel = ezsim_sim.rigid_solver.dofs_state.cdofd_vel.to_numpy()[:, 0]
    assert_allclose(ezsim_cdof_dot_vel[ezsim_dofs_idx], mj_cdof_dot_vel[mj_dofs_idx], tol=tol)

    # cinr
    ezsim_cinr_inertial = ezsim_sim.rigid_solver.links_state.cinr_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[:, :6]  # upper-triangular part
    assert_allclose(ezsim_cinr_inertial[ezsim_bodies_idx], mj_cinr_inertial[mj_bodies_idx], tol=tol)
    ezsim_cinr_pos = ezsim_sim.rigid_solver.links_state.cinr_pos.to_numpy()[:, 0]
    mj_cinr_pos = mj_sim.data.cinert[:, 6:9]
    assert_allclose(ezsim_cinr_pos[ezsim_bodies_idx], mj_cinr_pos[mj_bodies_idx], tol=tol)
    ezsim_cinr_mass = ezsim_sim.rigid_solver.links_state.cinr_mass.to_numpy()[:, 0]
    mj_cinr_mass = mj_sim.data.cinert[:, 9]
    assert_allclose(ezsim_cinr_mass[ezsim_bodies_idx], mj_cinr_mass[mj_bodies_idx], tol=tol)


def simulate_and_check_mujoco_consistency(ezsim_sim, mj_sim, qpos=None, qvel=None, *, tol, num_steps):
    # Get mapping between Mujoco and Genesis
    _, (_, _, mj_qs_idx, mj_dofs_idx, _, _) = _get_model_mappings(ezsim_sim, mj_sim)

    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(ezsim_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(ezsim_sim, mj_sim, qpos, qvel)

    # Run the simulation for a few steps
    qvel_prev = None

    for i in range(num_steps):
        # Make sure that all "dynamic" quantities are matching before stepping
        check_mujoco_data_consistency(ezsim_sim, mj_sim, qvel_prev=qvel_prev, tol=tol)

        # Keep Mujoco and Genesis simulation in sync to avoid drift over time
        mj_sim.data.qpos[mj_qs_idx] = ezsim_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = ezsim_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mj_sim.data.qacc_warmstart[mj_dofs_idx] = ezsim_sim.rigid_solver.constraint_solver.qacc_ws.to_numpy()[:, 0]
        mj_sim.data.qacc_smooth[mj_dofs_idx] = ezsim_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]

        # Backup current velocity
        qvel_prev = ezsim_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

        # Do a single simulation step (eventually with substeps for Genesis)
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        ezsim_sim.scene.step()
        # if ezsim_sim.scene.visualizer:
        #     ezsim_sim.scene.visualizer.update()
