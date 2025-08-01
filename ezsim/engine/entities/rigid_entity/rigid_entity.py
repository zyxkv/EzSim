from copy import copy
from itertools import chain
from typing import Literal

import numpy as np
import taichi as ti
import torch
import trimesh

import ezsim
from ezsim.engine.materials.base import Material
from ezsim.options.morphs import Morph
from ezsim.options.surfaces import Surface
from ezsim.utils import geom as gu
from ezsim.utils import linalg as lu
from ezsim.utils import mesh as mu
from ezsim.utils import mjcf as mju
from ezsim.utils import terrain as tu
from ezsim.utils import urdf as uu
from ezsim.utils.misc import tensor_to_array, ti_field_to_torch, ALLOCATE_TENSOR_WARNING

from ..base_entity import Entity
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import RigidLink


@ti.data_oriented
class RigidEntity(Entity):
    """
    Entity class in rigid body systems. One rigid entity can be a robot, a terrain, a floating rigid body, etc.
    """

    def __init__(
        self,
        scene: "Scene",
        solver: "Solver",
        material: Material,
        morph: Morph,
        surface: Surface,
        idx: int,
        idx_in_solver,
        link_start=0,
        joint_start=0,
        q_start=0,
        dof_start=0,
        geom_start=0,
        cell_start=0,
        vert_start=0,
        verts_state_start=0,
        face_start=0,
        edge_start=0,
        vgeom_start=0,
        vvert_start=0,
        vface_start=0,
        equality_start=0,
        visualize_contact=False,
    ):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._idx_in_solver = idx_in_solver
        self._link_start = link_start
        self._joint_start = joint_start
        self._q_start = q_start
        self._dof_start = dof_start
        self._geom_start = geom_start
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._verts_state_start = verts_state_start
        self._vgeom_start = vgeom_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._equality_start = equality_start

        self._base_links_idx = torch.tensor([self.base_link_idx], dtype=ezsim.tc_int, device=ezsim.device)

        self._visualize_contact = visualize_contact

        self._is_free = morph.is_free

        self._is_built = False

        self._load_model()

    def _load_model(self):
        self._links = ezsim.List()
        self._joints = ezsim.List()
        self._equalities = ezsim.List()

        if isinstance(self._morph, ezsim.morphs.Mesh):
            self._load_mesh(self._morph, self._surface)
        elif isinstance(self._morph, (ezsim.morphs.MJCF, ezsim.morphs.URDF, ezsim.morphs.Drone)):
            self._load_scene(self._morph, self._surface)
        elif isinstance(self._morph, ezsim.morphs.Primitive):
            self._load_primitive(self._morph, self._surface)
        elif isinstance(self._morph, ezsim.morphs.Terrain):
            self._load_terrain(self._morph, self._surface)
        else:
            ezsim.raise_exception(f"Unsupported morph: {self._morph}.")

        self._requires_jac_and_IK = self._morph.requires_jac_and_IK

        self._update_child_idxs()

    def _update_child_idxs(self):
        for link in self._links:
            if link.parent_idx != -1:
                parent_link = self._links[link.parent_idx_local]
                if link.idx not in parent_link.child_idxs:
                    parent_link.child_idxs.append(link.idx)

    def _load_primitive(self, morph, surface):
        if morph.fixed:
            joint_type = ezsim.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.zeros(0)
        else:
            joint_type = ezsim.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])

        if isinstance(morph, ezsim.options.morphs.Box):
            extents = np.array(morph.size)
            tmesh = mu.create_box(extents=extents)
            cmesh = tmesh
            geom_data = extents
            geom_type = ezsim.GEOM_TYPE.BOX
            link_name_prefix = "box"

        elif isinstance(morph, ezsim.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
            cmesh = tmesh
            geom_data = np.array([morph.radius])
            geom_type = ezsim.GEOM_TYPE.SPHERE
            link_name_prefix = "sphere"

        elif isinstance(morph, ezsim.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
            cmesh = tmesh
            geom_data = None
            geom_type = ezsim.GEOM_TYPE.MESH
            link_name_prefix = "cylinder"

        elif isinstance(morph, ezsim.options.morphs.Plane):
            tmesh, cmesh = mu.create_plane(normal=morph.normal)
            geom_data = np.array(morph.normal)
            geom_type = ezsim.GEOM_TYPE.PLANE
            link_name_prefix = "plane"

        else:
            ezsim.raise_exception("Unsupported primitive shape")

        # contains one visual geom (vgeom) and one collision geom (geom)
        g_infos = []
        if morph.visualization:
            g_infos.append(
                dict(
                    contype=0,
                    conaffinity=0,
                    vmesh=ezsim.Mesh.from_trimesh(tmesh, surface=surface),
                )
            )
        if morph.collision:
            g_infos.append(
                dict(
                    # contype=1,
                    # conaffinity=1,
                    contype=morph.contype,
                    conaffinity=morph.conaffinity,
                    mesh=ezsim.Mesh.from_trimesh(cmesh, surface=ezsim.surfaces.Collision()),
                    type=geom_type,
                    data=geom_data,
                    sol_params=gu.default_solver_params(),
                )
            )

        link, (joint,) = self._add_by_info(
            l_info=dict(
                is_robot=False,
                name=f"{link_name_prefix}_baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=gu.zero_pos(),
                inertial_quat=gu.identity_quat(),
                parent_idx=-1,
            ),
            j_infos=[
                dict(
                    name=f"{link_name_prefix}_baselink_joint",
                    n_qs=n_qs,
                    n_dofs=n_dofs,
                    type=joint_type,
                    init_qpos=init_qpos,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )

    def _load_mesh(self, morph, surface):
        if morph.fixed:
            joint_type = ezsim.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.zeros(0)
        else:
            joint_type = ezsim.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])

        # Load meshes
        meshes = ezsim.Mesh.from_morph_surface(morph, surface)

        g_infos = []
        if morph.visualization:
            for mesh in meshes:
                g_infos.append(
                    dict(
                        contype=0,
                        conaffinity=0,
                        vmesh=mesh,
                    )
                )
        if morph.collision:
            # Merge them as a single one if requested
            if morph.merge_submeshes_for_collision and len(meshes) > 1:
                tmesh = trimesh.util.concatenate([mesh.trimesh for mesh in meshes])
                mesh = ezsim.Mesh.from_trimesh(
                    mesh=tmesh,
                    surface=ezsim.surfaces.Collision(),
                )
                meshes = (mesh,)

            for mesh in meshes:
                g_infos.append(
                    dict(
                        # contype=1,
                        # conaffinity=1,
                        contype=morph.contype,
                        conaffinity=morph.conaffinity,
                        mesh=mesh,
                        type=ezsim.GEOM_TYPE.MESH,
                        sol_params=gu.default_solver_params(),
                    )
                )

        link_name = morph.file.rsplit("/", 1)[-1].replace(".", "_")

        link, (joint,) = self._add_by_info(
            l_info=dict(
                is_robot=False,
                name=f"{link_name}_baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=gu.zero_pos(),
                inertial_quat=gu.identity_quat(),
                parent_idx=-1,
            ),
            j_infos=[
                dict(
                    name=f"{link_name}_baselink_joint",
                    n_qs=n_qs,
                    n_dofs=n_dofs,
                    type=joint_type,
                    init_qpos=init_qpos,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )

    def _load_terrain(self, morph, surface):
        vmesh, mesh, self.terrain_hf = tu.parse_terrain(morph, surface)
        self.terrain_scale = np.array((morph.horizontal_scale, morph.vertical_scale), dtype=ezsim.np_float)

        g_infos = []
        if morph.visualization:
            g_infos.append(
                dict(
                    contype=0,
                    conaffinity=0,
                    vmesh=vmesh,
                )
            )
        if morph.collision:
            g_infos.append(
                dict(
                    contype=1,
                    conaffinity=1,
                    mesh=mesh,
                    type=ezsim.GEOM_TYPE.TERRAIN,
                    sol_params=gu.default_solver_params(),
                )
            )

        link, (joint,) = self._add_by_info(
            l_info=dict(
                is_robot=False,
                name="baselink",
                pos=np.array(morph.pos),
                quat=np.array(morph.quat),
                inertial_pos=None,
                inertial_quat=gu.identity_quat(),
                inertial_i=None,
                inertial_mass=None,
                parent_idx=-1,
                invweight=None,
            ),
            j_infos=[
                dict(
                    name="joint_baselink",
                    n_qs=0,
                    n_dofs=0,
                    type=ezsim.JOINT_TYPE.FIXED,
                )
            ],
            g_infos=g_infos,
            morph=morph,
            surface=surface,
        )

    def _load_scene(self, morph, surface):
        # Mujoco's unified MJCF+URDF parser is not good enough for now to be used for loading both MJCF and URDF files.
        # First, it would happen when loading visual meshes having supported format (i.e. Collada files '.dae').
        # Second, it does not take into account URDF 'mimic' joint constraints. However, it does a better job at
        # initialized undetermined physics parameters.
        if isinstance(morph, ezsim.morphs.MJCF):
            # Mujoco's unified MJCF+URDF parser systematically for MJCF files
            l_infos, links_j_infos, links_g_infos, eqs_info = mju.parse_xml(morph, surface)
        else:
            # Custom "legacy" URDF parser for loading geometries (visual and collision) and equality constraints.
            # This is necessary because Mujoco cannot parse visual geometries (meshes) reliably for URDF.
            l_infos, links_j_infos, links_g_infos, eqs_info = uu.parse_urdf(morph, surface)

            # Mujoco's unified MJCF+URDF parser for only link, joints, and collision geometries properties.
            morph_ = copy(morph)
            morph_.visualization = False
            try:
                # Mujoco's unified MJCF+URDF parser for URDF files.
                # Note that Mujoco URDF parser completely ignores equality constraints.
                l_infos, links_j_infos_mj, links_g_infos_mj, _ = mju.parse_xml(morph_, surface)

                # Mujoco is not parsing actuators properties
                for j_info_gs in chain.from_iterable(links_j_infos):
                    for j_info_mj in chain.from_iterable(links_j_infos_mj):
                        if j_info_mj["name"] == j_info_gs["name"]:
                            for name in ("dofs_force_range", "dofs_armature", "dofs_kp", "dofs_kv"):
                                j_info_mj[name] = j_info_gs[name]
                links_j_infos = links_j_infos_mj

                # Take into account 'world' body if it was added automatically for our legacy URDF parser
                if len(links_g_infos_mj) == len(links_g_infos) + 1:
                    assert not links_g_infos_mj[0]
                    links_g_infos.insert(0, [])
                assert len(links_g_infos_mj) == len(links_g_infos)

                # Update collision geometries, ignoring fake" visual geometries returned by Mujoco, (which is using
                # collision as visual to avoid loading mesh files), and keeping the true visual geometries provided
                # by our custom legacy URDF parser.
                # Note that the Kinematic tree ordering is stable between Mujoco and Genesis (Hopefully!).
                for link_g_infos, link_g_infos_mj in zip(links_g_infos, links_g_infos_mj):
                    # Remove collision geometries from our legacy URDF parser
                    for i_g, g_info in tuple(enumerate(link_g_infos))[::-1]:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            del link_g_infos[i_g]

                    # Add visual geometries from Mujoco's unified MJCF+URDF parser
                    for g_info in link_g_infos_mj:
                        is_col = g_info["contype"] or g_info["conaffinity"]
                        if is_col:
                            link_g_infos.append(g_info)
            except (ValueError, AssertionError):
                ezsim.logger.info("Falling back to legacy URDF parser. Default values of physics properties may be off.")

        # Add free floating joint at root if necessary
        if (
            (isinstance(morph, ezsim.morphs.Drone) or (isinstance(morph, ezsim.morphs.URDF) and not morph.fixed))
            and links_j_infos
            and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0
        ):
            # Select the second joint down the kinematic tree if possible without messing up with fixed links to keep
            root_idx = 0
            for idx, (l_info, link_j_infos) in tuple(enumerate(zip(l_infos, links_j_infos)))[:2]:
                if (
                    l_info["name"] not in morph.links_to_keep
                    and l_info["parent_idx"] in (0, -1)
                    and sum(j_info["n_dofs"] for j_info in link_j_infos) == 0
                ):
                    root_idx = idx
                    continue
                break

            # Define free joint
            j_info = dict()
            j_info["name"] = "root_joint"
            j_info["type"] = ezsim.JOINT_TYPE.FREE
            j_info["n_qs"] = 7
            j_info["n_dofs"] = 6
            j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])
            j_info["pos"] = gu.zero_pos()
            j_info["quat"] = gu.identity_quat()
            j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            j_info["dofs_motion_vel"] = np.eye(6, 3)
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            j_info["dofs_stiffness"] = np.zeros(6)
            j_info["dofs_invweight"] = np.zeros(6)
            j_info["dofs_damping"] = np.zeros(6)
            if isinstance(morph, ezsim.morphs.Drone):
                mass_tot = sum(l_info["inertial_mass"] for l_info in l_infos)
                j_info["dofs_damping"][3:] = mass_tot * morph.default_base_ang_damping_scale
            j_info["dofs_armature"] = np.zeros(6)
            j_info["dofs_kp"] = np.zeros((6,), dtype=ezsim.np_float)
            j_info["dofs_kv"] = np.zeros((6,), dtype=ezsim.np_float)
            j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (6, 1))
            links_j_infos[root_idx] = [j_info]

            # Rename root link for clarity if relevant
            if root_idx == 0:
                l_infos[root_idx]["name"] = "base"

            # Shift root idx for all child links and replace root if no longer fixed wrt world
            for i_l in range(root_idx, len(l_infos)):
                l_info = l_infos[i_l]
                if "root_idx" in l_info and l_info["root_idx"] in (root_idx + 1, i_l):
                    l_info["root_idx"] = root_idx

            # Must invalidate invweight for all child links and joints because the root joint was fixed when it was
            # initially computed. Re-initialize it to some strictly negative value to trigger recomputation in solver.
            for i_l in range(root_idx, len(l_infos)):
                l_infos[i_l]["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in links_j_infos[i_l]:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Remove the world link if "useless", i.e. free or fixed joint without any geometry attached
        if not links_g_infos[0] and sum(j_info["n_dofs"] for j_info in links_j_infos[0]) == 0:
            del l_infos[0], links_j_infos[0], links_g_infos[0]
            for l_info in l_infos:
                l_info["parent_idx"] = max(l_info["parent_idx"] - 1, -1)
                if "root_idx" in l_info:
                    l_info["root_idx"] = max(l_info["root_idx"] - 1, -1)
        # URDF is a robot description file so all links have same root_idx
        if isinstance(morph, ezsim.morphs.URDF) and not morph._enable_mujoco_compatibility:
            for l_info in l_infos:
                l_info["root_idx"] = 0
        # Genesis requires links associated with free joints to be attached to the world directly
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if all(j_info["type"] == ezsim.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["parent_idx"] = -1

        # Force recomputing inertial information based on geometry if ill-defined for some reason
        is_inertia_invalid = False
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not all(j_info["type"] == ezsim.JOINT_TYPE.FIXED for j_info in link_j_infos) and (
                (l_info.get("inertial_mass") is None or l_info["inertial_mass"] <= 0.0)
                or (l_info.get("inertial_i") is None or np.diag(l_info["inertial_i"]) <= 0.0).any()
            ):
                if l_info.get("inertial_mass") is not None or l_info.get("inertial_i") is not None:
                    ezsim.logger.debug(
                        f"Invalid or undefined inertia for link '{l_info['name']}'. Force recomputing it based on geometry."
                    )
                l_info["inertial_i"] = None
                l_info["inertial_mass"] = None
                is_inertia_invalid = True
        if is_inertia_invalid:
            for l_info, link_j_infos in zip(l_infos, links_j_infos):
                l_info["invweight"] = np.full((2,), fill_value=-1.0)
                for j_info in link_j_infos:
                    j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)

        # Define a flag that determines whether the link at hand is associated with a robot.
        # Note that 0d array is used rather than native type because this algo requires mutable objects.
        for l_info, link_j_infos in zip(l_infos, links_j_infos):
            if not link_j_infos or all(j_info["type"] == ezsim.JOINT_TYPE.FIXED for j_info in link_j_infos):
                if l_info["parent_idx"] >= 0:
                    l_info["is_robot"] = l_infos[l_info["parent_idx"]]["is_robot"]
                else:
                    l_info["is_robot"] = np.array(False, dtype=np.bool_)
            elif all(j_info["type"] == ezsim.JOINT_TYPE.FREE for j_info in link_j_infos):
                l_info["is_robot"] = np.array(False, dtype=np.bool_)
            else:
                l_info["is_robot"] = np.array(True, dtype=np.bool_)
                if l_info["parent_idx"] >= 0:
                    l_infos[l_info["parent_idx"]]["is_robot"][()] = True

        # Add (link, joints, geoms) tuples sequentially
        for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
            if l_info["parent_idx"] < 0:
                if morph.pos is not None or morph.quat is not None:
                    ezsim.logger.info("Applying offset to base link's pose with user provided value in morph.")
                    pos = np.asarray(l_info.get("pos", (0.0, 0.0, 0.0)))
                    quat = np.asarray(l_info.get("quat", (1.0, 0.0, 0.0, 0.0)))
                    if morph.pos is None:
                        pos_offset = np.zeros((3,))
                    else:
                        pos_offset = np.asarray(morph.pos)
                    if morph.quat is None:
                        quat_offset = np.array((1.0, 0.0, 0.0, 0.0))
                    else:
                        quat_offset = np.asarray(morph.quat)
                    l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
                        pos, quat, pos_offset, quat_offset
                    )

                for j_info in link_j_infos:
                    if j_info["type"] == ezsim.JOINT_TYPE.FREE:
                        # in this case, l_info['pos'] and l_info['quat'] are actually not used in solver,
                        # but this initial value will be reflected
                        j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

            # Exclude joints with 0 dofs to align with Mujoco
            link_j_infos = [j_info for j_info in link_j_infos if j_info["n_dofs"] > 0]

            self._add_by_info(l_info, link_j_infos, link_g_infos, morph, surface)

        # Add equality constraints sequentially
        for eq_info in eqs_info:
            self._add_equality(
                name=eq_info["name"],
                type=eq_info["type"],
                objs_name=eq_info["objs_name"],
                data=eq_info["data"],
                sol_params=eq_info["sol_params"],
            )

    def _build(self):
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._is_built = True

        self._geoms = self.geoms
        self._vgeoms = self.vgeoms

        self._init_jac_and_IK()

    def _init_jac_and_IK(self):
        if not self._requires_jac_and_IK:
            return

        if self.n_dofs == 0:
            return

        self._jacobian = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape((6, self.n_dofs)))

        # compute joint limit in q space
        q_limit_lower = []
        q_limit_upper = []
        for joint in self.joints:
            if joint.type == ezsim.JOINT_TYPE.FREE:
                q_limit_lower.append(joint.dofs_limit[:3, 0])
                q_limit_lower.append(-np.ones(4))  # quaternion lower bound
                q_limit_upper.append(joint.dofs_limit[:3, 1])
                q_limit_upper.append(np.ones(4))  # quaternion upper bound
            elif joint.type == ezsim.JOINT_TYPE.FIXED:
                pass
            else:
                q_limit_lower.append(joint.dofs_limit[:, 0])
                q_limit_upper.append(joint.dofs_limit[:, 1])
        self.q_limit = np.stack(
            (np.concatenate(q_limit_lower), np.concatenate(q_limit_upper)), axis=0, dtype=ezsim.np_float
        )

        # for storing intermediate results
        self._IK_n_tgts = self._solver._options.IK_max_targets
        self._IK_error_dim = self._IK_n_tgts * 6
        self._IK_mat = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_inv = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_L = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_U = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_y = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_qpos_orig = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._IK_qpos_best = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._IK_delta_qpos = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self.n_dofs))
        self._IK_vec = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_err_pose = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_err_pose_best = ti.field(dtype=ezsim.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_jacobian = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self.n_dofs))
        )
        self._IK_jacobian_T = ti.field(
            dtype=ezsim.ti_float, shape=self._solver._batch_shape((self.n_dofs, self._IK_error_dim))
        )

    def _add_by_info(self, l_info, j_infos, g_infos, morph, surface):
        if len(j_infos) > 1 and any(j_info["type"] in (ezsim.JOINT_TYPE.FREE, ezsim.JOINT_TYPE.FIXED) for j_info in j_infos):
            raise ValueError(
                "Compounding joints of types 'FREE' or 'FIXED' with any other joint on the same body not supported"
            )

        if isinstance(morph, ezsim.options.morphs.FileMorph) and morph.recompute_inertia:
            l_info.update(inertial_pos=None, inertial_quat=None, inertial_i=None, inertial_mass=None)

        parent_idx = l_info["parent_idx"]
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_info.get("root_idx")
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start

        link = RigidLink(
            entity=self,
            name=l_info["name"],
            idx=self.n_links + self._link_start,
            joint_start=self.n_joints + self._joint_start,
            n_joints=len(j_infos),
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=l_info["pos"],
            quat=l_info["quat"],
            inertial_pos=l_info.get("inertial_pos"),
            inertial_quat=l_info.get("inertial_quat"),
            inertial_i=l_info.get("inertial_i"),
            inertial_mass=l_info.get("inertial_mass"),
            parent_idx=parent_idx,
            root_idx=root_idx,
            invweight=l_info.get("invweight"),
            visualize_contact=self.visualize_contact,
        )
        self._links.append(link)

        joints = ezsim.List()
        self._joints.append(joints)
        for j_info in j_infos:
            n_dofs = j_info["n_dofs"]

            sol_params = np.array(j_info.get("sol_params", gu.default_solver_params()), copy=True)
            if (
                len(sol_params.shape) == 2
                and sol_params.shape[0] == 1
                and (sol_params[0][3] >= 1.0 or sol_params[0][2] >= sol_params[0][3])
            ):
                ezsim.logger.warning(
                    f"Joint {j_info['name']}'s sol_params {sol_params[0]} look not right, change to default."
                )
                sol_params = gu.default_solver_params()

            dofs_motion_ang = j_info.get("dofs_motion_ang")
            if dofs_motion_ang is None:
                if n_dofs == 6:
                    dofs_motion_ang = np.eye(6, 3, -3)
                elif n_dofs == 0:
                    dofs_motion_ang = np.zeros((0, 3))
                else:
                    assert False

            dofs_motion_vel = j_info.get("dofs_motion_vel")
            if dofs_motion_vel is None:
                if n_dofs == 6:
                    dofs_motion_vel = np.eye(6, 3)
                elif n_dofs == 0:
                    dofs_motion_vel = np.zeros((0, 3))
                else:
                    assert False

            joint = RigidJoint(
                entity=self,
                name=j_info["name"],
                idx=self.n_joints + self._joint_start,
                link_idx=link.idx,
                q_start=self.n_qs + self._q_start,
                dof_start=self.n_dofs + self._dof_start,
                n_qs=j_info["n_qs"],
                n_dofs=n_dofs,
                type=j_info["type"],
                pos=j_info.get("pos", gu.zero_pos()),
                quat=j_info.get("quat", gu.identity_quat()),
                init_qpos=j_info.get("init_qpos", np.zeros(n_dofs)),
                sol_params=sol_params,
                dofs_motion_ang=dofs_motion_ang,
                dofs_motion_vel=dofs_motion_vel,
                dofs_limit=j_info.get("dofs_limit", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
                dofs_invweight=j_info.get("dofs_invweight", np.zeros(n_dofs)),
                dofs_stiffness=j_info.get("dofs_stiffness", np.zeros(n_dofs)),
                dofs_damping=j_info.get("dofs_damping", np.zeros(n_dofs)),
                dofs_armature=j_info.get("dofs_armature", np.zeros(n_dofs)),
                dofs_kp=j_info.get("dofs_kp", np.zeros(n_dofs)),
                dofs_kv=j_info.get("dofs_kv", np.zeros(n_dofs)),
                dofs_force_range=j_info.get("dofs_force_range", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
            )
            joints.append(joint)

        # Separate collision from visual geometry for post-processing
        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once.
        # Destroying the original geometries should be avoided if possible as it will change the way objects
        # interact with the world due to only computing one contact point per convex geometry. The idea is to
        # check if each geometry can be convexified independently without resorting on convex decomposition.
        # If so, the original geometries are preserve. If not, then they are all merged as one. Following the
        # same approach as before, the resulting geometry is convexify without resorting on convex decomposition
        # if possible. Mergeing before falling back directly to convex decompositio is important as it gives one
        # last chance to avoid it. Moreover, it tends to reduce the final number of collision geometries. In
        # both cases, this improves runtime performance, numerical stability and compilation time.
        if isinstance(morph, ezsim.options.morphs.FileMorph):
            # Choose the appropriate convex decomposition error threshold depending on whether the link at hand
            # is associated with a robot.
            # The rational behind it is that performing convex decomposition for robots is mostly useless because
            # the non-physical part that is added to the original geometries to convexify them are generally inside
            # the mechanical structure and not interacting directly with the outer world. On top of that, not only
            # iy increases the memory footprint and compilation time, but also the simulation speed (marginally).
            if l_info["is_robot"]:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            cg_infos = mu.postprocess_collision_geoms(
                cg_infos,
                morph.decimate,
                morph.decimate_face_num,
                morph.decimate_aggressiveness,
                morph.convexify,
                decompose_error_threshold,
                morph.coacd_options,
            )

        # Randomize collision mesh colors. The is especially useful to check convex decomposition.
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        # Add visual geometries
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )

        # Add collision geometries
        for g_info in cg_infos:
            friction = g_info.get("friction", self.material.friction)
            if friction is None:
                friction = gu.default_friction()
            link._add_geom(
                mesh=g_info["mesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
                type=g_info["type"],
                friction=friction,
                sol_params=g_info["sol_params"],
                data=g_info.get("data"),
                needs_coup=self.material.needs_coup,
                contype=g_info["contype"],
                conaffinity=g_info["conaffinity"],
            )

        return link, joints

    def _add_equality(self, name, type, objs_name, data, sol_params):
        objs_id = []
        for obj_name in objs_name:
            if type == ezsim.EQUALITY_TYPE.CONNECT:
                obj_id = self.get_link(obj_name).idx
            elif type == ezsim.EQUALITY_TYPE.JOINT:
                obj_id = self.get_joint(obj_name).idx
            elif type == ezsim.EQUALITY_TYPE.WELD:
                obj_id = self.get_link(obj_name).idx
            else:
                ezsim.logger.warning(f"Equality type {type} not supported. Only CONNECT, JOINT, and WELD are supported.")
            objs_id.append(obj_id)

        equality = RigidEquality(
            entity=self,
            name=name,
            idx=self.n_equalities + self._equality_start,
            type=type,
            eq_obj1id=objs_id[0],
            eq_obj2id=objs_id[1],
            eq_data=data,
            sol_params=sol_params,
        )
        self._equalities.append(equality)
        return equality

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    @ezsim.assert_built
    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link’s *local* frame.
            If None, the link origin is used (back-compat).

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        if not self._requires_jac_and_IK:
            ezsim.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            ezsim.raise_exception("Entity has zero dofs.")

        if local_point is None:
            self._kernel_get_jacobian_zero(link.idx)
        else:
            p_local = torch.as_tensor(local_point, dtype=ezsim.tc_float, device=ezsim.device)
            if p_local.shape != (3,):
                ezsim.raise_exception("Must be a vector of length 3")
            self._kernel_get_jacobian(link.idx, p_local)

        jacobian = self._jacobian.to_torch(ezsim.device).permute(2, 0, 1)
        if self._solver.n_envs == 0:
            jacobian = jacobian.squeeze(0)

        return jacobian

    @ti.func
    def _impl_get_jacobian(self, tgt_link_idx, i_b, p_vec):
        self._func_get_jacobian(
            tgt_link_idx,
            i_b,
            p_vec,
            ti.Vector.one(ezsim.ti_int, 3),
            ti.Vector.one(ezsim.ti_int, 3),
        )

    @ti.kernel
    def _kernel_get_jacobian(self, tgt_link_idx: ti.i32, p_local: ti.types.ndarray()):
        p_vec = ti.Vector([p_local[0], p_local[1], p_local[2]], dt=ezsim.ti_float)
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(tgt_link_idx, i_b, p_vec)

    @ti.kernel
    def _kernel_get_jacobian_zero(self, tgt_link_idx: ti.i32):
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(tgt_link_idx, i_b, ti.Vector.zero(ezsim.ti_float, 3))

    @ti.func
    def _func_get_jacobian(self, tgt_link_idx, i_b, p_local, pos_mask, rot_mask):
        for i_row, i_d in ti.ndrange(6, self.n_dofs):
            self._jacobian[i_row, i_d, i_b] = 0.0

        tgt_link_pos = self._solver.links_state.pos[tgt_link_idx, i_b] + gu.ti_transform_by_quat(
            p_local, self._solver.links_state.quat[tgt_link_idx, i_b]
        )
        i_l = tgt_link_idx
        while i_l > -1:
            I_l = [i_l, i_b] if ti.static(self.solver._options.batch_links_info) else i_l

            dof_offset = 0
            for i_j in range(self._solver.links_info.joint_start[I_l], self._solver.links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(self.solver._options.batch_joints_info) else i_j

                if self._solver.joints_info.type[I_j] == ezsim.JOINT_TYPE.FIXED:
                    pass

                elif self._solver.joints_info.type[I_j] == ezsim.JOINT_TYPE.REVOLUTE:
                    i_d = self._solver.joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    rotation = gu.ti_transform_by_quat(
                        self._solver.dofs_info.motion_ang[I_d], self._solver.links_state.quat[i_l, i_b]
                    )
                    translation = rotation.cross(tgt_link_pos - self._solver.links_state.pos[i_l, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                    self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                    self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                    self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                elif self._solver.joints_info.type[I_j] == ezsim.JOINT_TYPE.PRISMATIC:
                    i_d = self._solver.joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    translation = gu.ti_transform_by_quat(
                        self._solver.dofs_info.motion_vel[I_d], self._solver.links_state.quat[i_l, i_b]
                    )

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]

                elif self._solver.joints_info.type[I_j] == ezsim.JOINT_TYPE.FREE:
                    # translation
                    for i_d_ in ti.static(range(3)):
                        i_d = self._solver.joints_info.dof_start[I_j] + i_d_
                        i_d_jac = i_d + dof_offset - self._dof_start

                        self._jacobian[i_d_, i_d_jac, i_b] = 1.0 * pos_mask[i_d_]

                    # rotation
                    for i_d_ in ti.static(range(3)):
                        i_d = self._solver.joints_info.dof_start[I_j] + i_d_ + 3
                        i_d_jac = i_d + dof_offset - self._dof_start
                        I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                        rotation = self._solver.dofs_info.motion_ang[I_d]
                        translation = rotation.cross(tgt_link_pos - self._solver.links_state.pos[i_l, i_b])

                        self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                        self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                        self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                        self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                        self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                        self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                dof_offset = dof_offset + self._solver.joints_info.n_dofs[I_j]

            i_l = self._solver.links_info.parent_idx[I_l]

    @ezsim.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single target link.

        Parameters
        ----------
        link : RigidLink
            The link to be used as the end-effector.
        pos : None | array_like, shape (3,), optional
            The target position. If None, position error will not be considered. Defaults to None.
        quat : None | array_like, shape (4,), optional
            The target orientation. If None, orientation error will not be considered. Defaults to None.
        init_qpos : None | array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx: None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            if pos is not None:
                if pos.shape[0] != len(envs_idx):
                    ezsim.raise_exception("First dimension of `pos` must be equal to `scene.n_envs`.")
            if quat is not None:
                if quat.shape[0] != len(envs_idx):
                    ezsim.raise_exception("First dimension of `quat` must be equal to `scene.n_envs`.")

        ret = self.inverse_kinematics_multilink(
            links=[link],
            poss=[pos] if pos is not None else [],
            quats=[quat] if quat is not None else [],
            init_qpos=init_qpos,
            respect_joint_limit=respect_joint_limit,
            max_samples=max_samples,
            max_solver_iters=max_solver_iters,
            damping=damping,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            pos_mask=pos_mask,
            rot_mask=rot_mask,
            max_step_size=max_step_size,
            dofs_idx_local=dofs_idx_local,
            return_error=return_error,
            envs_idx=envs_idx,
        )

        if return_error:
            qpos, error_pose = ret
            error_pose = error_pose.squeeze(-2)  # 1 single link
            return qpos, error_pose

        else:
            return ret

    @ezsim.assert_built
    def inverse_kinematics_multilink(
        self,
        links,
        poss=[],
        quats=[],
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for  multiple target links.

        Parameters
        ----------
        links : list of RigidLink
            List of links to be used as the end-effectors.
        poss : list, optional
            List of target positions. If empty, position error will not be considered. Defaults to None.
        quats : list, optional
            List of target orientations. If empty, orientation error will not be considered. Defaults to None.
        init_qpos : array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        if not self._requires_jac_and_IK:
            ezsim.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            ezsim.raise_exception("Entity has zero dofs.")

        n_links = len(links)
        if n_links == 0:
            ezsim.raise_exception("Target link not provided.")

        if len(poss) == 0:
            poss = [None for _ in range(n_links)]
            pos_mask = [False, False, False]
        elif len(poss) != n_links:
            ezsim.raise_exception("Accepting only `poss` with length equal to `links` or empty list.")

        if len(quats) == 0:
            quats = [None for _ in range(n_links)]
            rot_mask = [False, False, False]
        elif len(quats) != n_links:
            ezsim.raise_exception("Accepting only `quats` with length equal to `links` or empty list.")

        link_pos_mask = []
        link_rot_mask = []
        for i in range(n_links):
            if poss[i] is None and quats[i] is None:
                ezsim.raise_exception("At least one of `poss` or `quats` must be provided.")
            if poss[i] is not None:
                link_pos_mask.append(True)
                if self._solver.n_envs > 0:
                    if poss[i].shape[0] != len(envs_idx):
                        ezsim.raise_exception("First dimension of elements in `poss` must be equal to scene.n_envs.")
            else:
                link_pos_mask.append(False)
                if self._solver.n_envs == 0:
                    poss[i] = gu.zero_pos()
                else:
                    poss[i] = self._solver._batch_array(gu.zero_pos(), True)
            if quats[i] is not None:
                link_rot_mask.append(True)
                if self._solver.n_envs > 0:
                    if quats[i].shape[0] != len(envs_idx):
                        ezsim.raise_exception("First dimension of elements in `quats` must be equal to scene.n_envs.")
            else:
                link_rot_mask.append(False)
                if self._solver.n_envs == 0:
                    quats[i] = gu.identity_quat()
                else:
                    quats[i] = self._solver._batch_array(gu.identity_quat(), True)

        if init_qpos is not None:
            init_qpos = torch.as_tensor(init_qpos, dtype=ezsim.tc_float)
            if init_qpos.shape[-1] != self.n_qs:
                ezsim.raise_exception(
                    f"Size of last dimension `init_qpos` does not match entity's `n_qs`: {init_qpos.shape[-1]} vs {self.n_qs}."
                )

            init_qpos = self._solver._process_dim(init_qpos)
            custom_init_qpos = True

        else:
            init_qpos = torch.empty((0, 0), dtype=ezsim.tc_float)  # B * n_qs, dummy
            custom_init_qpos = False

        # pos and rot mask
        pos_mask = torch.as_tensor(pos_mask, dtype=bool, device=ezsim.device)
        if len(pos_mask) != 3:
            ezsim.raise_exception("`pos_mask` must have length 3.")
        rot_mask = torch.as_tensor(rot_mask, dtype=bool, device=ezsim.device)
        if len(rot_mask) != 3:
            ezsim.raise_exception("`rot_mask` must have length 3.")
        if sum(rot_mask) == 1:
            rot_mask = ~rot_mask
        elif sum(rot_mask) == 2:
            ezsim.raise_exception("You can only align 0, 1 axis or all 3 axes.")
        else:
            pass  # nothing needs to change for 0 or 3 axes
        link_pos_mask = torch.as_tensor(link_pos_mask, dtype=ezsim.tc_int, device=ezsim.device)
        link_rot_mask = torch.as_tensor(link_rot_mask, dtype=ezsim.tc_int, device=ezsim.device)

        links_idx = torch.as_tensor([link.idx for link in links], dtype=ezsim.tc_int, device=ezsim.device)
        poss = torch.stack(
            [
                self._solver._process_dim(torch.as_tensor(pos, dtype=ezsim.tc_float, device=ezsim.device), envs_idx=envs_idx)
                for pos in poss
            ]
        )
        quats = torch.stack(
            [
                self._solver._process_dim(torch.as_tensor(quat, dtype=ezsim.tc_float, device=ezsim.device), envs_idx=envs_idx)
                for quat in quats
            ]
        )

        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, unsafe=False)
        n_dofs = len(dofs_idx)
        if n_dofs == 0:
            ezsim.raise_exception("Target dofs not provided.")
        links_idx_by_dofs = []
        for link in self.links:
            for joint in link.joints:
                if any(i in dofs_idx for i in joint.dofs_idx_local):
                    links_idx_by_dofs.append(link.idx_local)  # converted to global later
                    break

        links_idx_by_dofs = self._get_idx(links_idx_by_dofs, self.n_links, self._link_start, unsafe=False)
        n_links_by_dofs = len(links_idx_by_dofs)

        if envs_idx is None:
            envs_idx = torch.zeros(1, dtype=ezsim.tc_int, device=ezsim.device)

        self._solver.rigid_entity_inverse_kinematics(
            links_idx,
            poss,
            quats,
            n_links,
            dofs_idx,
            n_dofs,
            links_idx_by_dofs,
            n_links_by_dofs,
            custom_init_qpos,
            init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            pos_mask,
            rot_mask,
            link_pos_mask,
            link_rot_mask,
            max_step_size,
            respect_joint_limit,
            envs_idx,
            self,
        )

        qpos = self._IK_qpos_best.to_torch(ezsim.device).transpose(1, 0)
        if self._solver.n_envs > 0:
            qpos = qpos[envs_idx]
        else:
            qpos = qpos.squeeze(0)

        if return_error:
            error_pose = (
                self._IK_err_pose_best.to_torch(ezsim.device).reshape((self._IK_n_tgts, 6, -1))[:n_links].permute(2, 0, 1)
            )
            if self._solver.n_envs == 0:
                error_pose = error_pose.squeeze(0)
            return qpos, error_pose
        return qpos

    @ezsim.assert_built
    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """
        Compute forward kinematics for a single target link.

        Parameters
        ----------
        qpos : array_like, shape (n_qs,) or (n_envs, n_qs) or (len(envs_idx), n_qs)
            The joint positions.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Defaults to None.
        links_idx_local : None | array_like, optional
            The indices of the links to get. If None, all links will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        links_pos : array_like, shape (n_links, 3) or (n_envs, n_links, 3) or (len(envs_idx), n_links, 3)
            The positions of the links (link frame origins).
        links_quat : array_like, shape (n_links, 4) or (n_envs, n_links, 4) or (len(envs_idx), n_links, 4)
            The orientations of the links.
        """

        if self._solver.n_envs == 0:
            qpos = qpos.unsqueeze(0)
            envs_idx = torch.zeros(1, dtype=ezsim.tc_int)
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=False)
        links_pos = torch.empty((len(envs_idx), len(links_idx), 3), dtype=ezsim.tc_float, device=ezsim.device)
        links_quat = torch.empty((len(envs_idx), len(links_idx), 4), dtype=ezsim.tc_float, device=ezsim.device)

        self._kernel_forward_kinematics(
            links_pos,
            links_quat,
            qpos,
            self._get_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=False),
            links_idx,
            envs_idx,
        )

        if self._solver.n_envs == 0:
            links_pos = links_pos.squeeze(0)
            links_quat = links_quat.squeeze(0)
        return links_pos, links_quat

    @ti.kernel
    def _kernel_forward_kinematics(
        self,
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        qpos: ti.types.ndarray(),
        qs_idx: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < ezsim.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # save original qpos
            # NOTE: reusing the IK_qpos_orig as cache (should not be a problem)
            self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]] = self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]]
            # set new qpos
            self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]
            # run FK
            self._solver._func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                self._solver.links_state,
                self._solver.links_info,
                self._solver.joints_state,
                self._solver.joints_info,
                self._solver.dofs_state,
                self._solver.dofs_info,
                self._solver.entities_info,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
            )

        ti.loop_config(serialize=self._solver._para_level < ezsim.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                links_pos[i_b_, i_l_, i] = self._solver.links_state.pos[links_idx[i_l_], envs_idx[i_b_]]
            for i in ti.static(range(4)):
                links_quat[i_b_, i_l_, i] = self._solver.links_state.quat[links_idx[i_l_], envs_idx[i_b_]]

        ti.loop_config(serialize=self._solver._para_level < ezsim.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # restore original qpos
            self._solver.qpos[qs_idx[i_q_], envs_idx[i_b_]] = self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]]
            # run FK
            self._solver._func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                self._solver.links_state,
                self._solver.links_info,
                self._solver.joints_state,
                self._solver.joints_info,
                self._solver.dofs_state,
                self._solver.dofs_info,
                self._solver.entities_info,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
            )

    # ------------------------------------------------------------------------------------
    # --------------------------------- motion planing -----------------------------------
    # ------------------------------------------------------------------------------------
    @ezsim.assert_built
    def plan_path(
        self,
        qpos_goal,
        qpos_start=None,
        max_nodes=2000,
        resolution=0.05,
        timeout=None,
        max_retry=1,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        planner="RRTConnect",
        envs_idx=None,
        return_valid_mask=False,
        *,
        ee_link_name=None,
        with_entity=None,
        **kwargs,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used.
            Defaults to None. [B, Nq] or [1, Nq]
        resolution : float, optiona
            Joint-space resolution. It corresponds to the maximum distance between states to be checked
            for validity along a path segment.
        timeout : float, optional
            The max time to spend for each planning in seconds. Note that the timeout is not exact.
        max_retry : float, optional
            Maximum number of retry in case of timeout or convergence failure. Default to 1.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed.
            Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            This option has been deprecated and is not longer doing anything.
        planner : str, optional
            The name of the motion planning algorithm to use.
            Supported planners: 'RRT', 'RRTConnect'. Defaults to 'RRTConnect'.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        return_valid_mask: bool
            Obtain valid mask of the succesful planed path over batch.
        ee_link_name: str
            The name of the link, which we "attach" the object during the planning
        with_entity: RigidEntity
            The (non-articulated) object to "attach" during the planning

        Returns
        -------
        path : torch.Tensor
            A tensor of waypoints representing the planned path.
            Each waypoint is an array storing the entity's qpos of a single time step.
        is_invalid: torch.Tensor
            A tensor of boolean mask indicating the batch indices with failed plan.
        """
        if self._solver.n_envs > 0:
            n_envs = len(self._scene._sanitize_envs_idx(envs_idx))
        else:
            n_envs = 1

        if "ignore_joint_limit" in kwargs:
            ezsim.logger.warning("`ignore_joint_limit` is deprecated")

        ee_link_idx = None
        if ee_link_name is not None:
            assert with_entity is not None, "`with_entity` must be specified."
            ee_link_idx = self.get_link(ee_link_name).idx
        if with_entity is not None:
            assert ee_link_name is not None, "reference link of the robot must be specified."
            assert len(with_entity.links) == 1, "only non-articulated object is supported for now."

        # import here to avoid circular import
        from ezsim.utils.path_planing import RRT, RRTConnect

        match planner:
            case "RRT":
                planner_obj = RRT(self)
            case "RRTConnect":
                planner_obj = RRTConnect(self)
            case _:
                ezsim.raise_exception(f"invalid planner {planner} specified.")

        path = torch.empty((num_waypoints, n_envs, self.n_qs), dtype=ezsim.tc_float, device=ezsim.device)
        is_invalid = torch.ones((n_envs,), dtype=torch.bool, device=ezsim.device)
        for i in range(1 + max_retry):
            retry_path, retry_is_invalid = planner_obj.plan(
                qpos_goal,
                qpos_start=qpos_start,
                resolution=resolution,
                timeout=timeout,
                max_nodes=max_nodes,
                smooth_path=smooth_path,
                num_waypoints=num_waypoints,
                ignore_collision=ignore_collision,
                envs_idx=envs_idx,
                ee_link_idx=ee_link_idx,
                obj_entity=with_entity,
            )
            # NOTE: update the previously failed path with the new results
            path[:, is_invalid] = retry_path[:, is_invalid]

            is_invalid &= retry_is_invalid
            if not is_invalid.any():
                break
            ezsim.logger.info(f"Planning failed. Retrying for {is_invalid.sum()} environments...")

        if self._solver.n_envs == 0:
            if return_valid_mask:
                return path.squeeze(1), is_invalid[0]
            return path.squeeze(1)

        if return_valid_mask:
            return path, is_invalid
        return path

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_joint(self, name=None, uid=None):
        """
        Get a RigidJoint object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the joint. Defaults to None.
        uid : str, optional
            The uid of the joint. This can be a substring of the joint's uid. Defaults to None.

        Returns
        -------
        joint : RigidJoint
            The joint object.
        """

        if name is not None:
            for joint in self.joints:
                if joint.name == name:
                    return joint
            ezsim.raise_exception(f"Joint not found for name: {name}.")

        elif uid is not None:
            for joint in self.joints:
                if uid in str(joint.uid):
                    return joint
            ezsim.raise_exception(f"Joint not found for uid: {uid}.")

        else:
            ezsim.raise_exception("Neither `name` nor `uid` is provided.")

    def get_link(self, name=None, uid=None):
        """
        Get a RigidLink object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the link. Defaults to None.
        uid : str, optional
            The uid of the link. This can be a substring of the link's uid. Defaults to None.

        Returns
        -------
        link : RigidLink
            The link object.
        """

        if name is not None:
            for link in self._links:
                if link.name == name:
                    return link
            ezsim.raise_exception(f"Link not found for name: {name}.")

        elif uid is not None:
            for link in self._links:
                if uid in str(link.uid):
                    return link
            ezsim.raise_exception(f"Link not found for uid: {uid}.")

        else:
            ezsim.raise_exception("Neither `name` nor `uid` is provided.")

    @ezsim.assert_built
    def get_pos(self, envs_idx=None, *, unsafe=False):
        """
        Returns position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The position of the entity's base link.
        """
        return self._solver.get_links_pos(self._base_links_idx, envs_idx, unsafe=unsafe).squeeze(-2)

    @ezsim.assert_built
    def get_quat(self, envs_idx=None, *, unsafe=False):
        """
        Returns quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The quaternion of the entity's base link.
        """
        return self._solver.get_links_quat(self._base_links_idx, envs_idx, unsafe=unsafe).squeeze(-2)

    @ezsim.assert_built
    def get_vel(self, envs_idx=None, *, unsafe=False):
        """
        Returns linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (3,) or (n_envs, 3)
            The linear velocity of the entity's base link.
        """
        return self._solver.get_links_vel(self._base_links_idx, envs_idx, unsafe=unsafe).squeeze(-2)

    @ezsim.assert_built
    def get_ang(self, envs_idx=None, *, unsafe=False):
        """
        Returns angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (3,) or (n_envs, 3)
            The angular velocity of the entity's base link.
        """
        return self._solver.get_links_ang(self._base_links_idx, envs_idx, unsafe=unsafe).squeeze(-2)

    @ezsim.assert_built
    def get_links_pos(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Returns position of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_quat(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_vel(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com"] = "link_origin",
        unsafe=False,
    ):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com"
            The reference point being used to expressed the velocity of each link.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx, ref=ref, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_ang(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Returns angular velocity of all the entity's links in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The angular velocity of all the entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_ang(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_accelerometer_data(self, links_idx_local=None, envs_idx=None, *, imu=False, unsafe=False):
        """
        Returns the accelerometer data that would be measured by a IMU rigidly attached to the specified entity's links,
        i.e. the true linear acceleration of the links expressed at their respective origin in local frame coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The accelerometer data of IMUs rigidly attached of the specified entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx, mimick_imu=True, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_acc(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Returns true linear acceleration (aka. "classical acceleration") of the specified entity's links expressed at
        their respective origin in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx, mimick_imu=False, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_acc_ang(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Returns angular acceleration of the specified entity's links expressed at their respective origin in world
        coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc_ang(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_inertial_mass(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_inertial_mass(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_links_invweight(self, links_idx_local=None, envs_idx=None, *, unsafe=False):
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_invweight(links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_pos(self, pos, envs_idx=None, *, relative=False, zero_velocity=True, unsafe=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        relative : bool, optional
            Whether the position to set is absolute or relative to the initial (not current!) position. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if not unsafe:
            _pos = torch.as_tensor(pos, dtype=ezsim.tc_float, device=ezsim.device).contiguous()
            if _pos is not pos:
                ezsim.logger.debug(ALLOCATE_TENSOR_WARNING)
            pos = _pos
        self._solver.set_base_links_pos(
            pos.unsqueeze(-2),
            self._base_links_idx,
            envs_idx,
            relative=relative,
            unsafe=unsafe,
            skip_forward=zero_velocity,
        )
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_quat(self, quat, envs_idx=None, *, relative=False, zero_velocity=True, unsafe=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        relative : bool, optional
            Whether the quaternion to set is absolute or relative to the initial (not current!) quaternion. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if not unsafe:
            _quat = torch.as_tensor(quat, dtype=ezsim.tc_float, device=ezsim.device).contiguous()
            if _quat is not quat:
                ezsim.logger.debug(ALLOCATE_TENSOR_WARNING)
            quat = _quat
        self._solver.set_base_links_quat(
            quat.unsqueeze(-2),
            self._base_links_idx,
            envs_idx,
            relative=relative,
            unsafe=unsafe,
            skip_forward=zero_velocity,
        )
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_verts(self):
        """
        Get the all vertices of the entity (using collision geoms).

        Returns
        -------
        verts : torch.Tensor, shape (n_verts, 3) or (n_envs, n_verts, 3)
            The vertices of the entity (using collision geoms).
        """
        self._update_verts_for_geom()
        if self.is_free:
            tensor = torch.empty(
                self._solver._batch_shape((self.n_verts, 3), True), dtype=ezsim.tc_float, device=ezsim.device
            )
            self._kernel_get_free_verts(tensor)
            if self._solver.n_envs == 0:
                tensor = tensor.squeeze(0)
        else:
            tensor = torch.empty((self.n_verts, 3), dtype=ezsim.tc_float, device=ezsim.device)
            self._kernel_get_fixed_verts(tensor)
        return tensor

    @ezsim.assert_built
    def _update_verts_for_geom(self):
        for i_g_ in range(self.n_geoms):
            i_g = i_g_ + self._geom_start
            self._solver.update_verts_for_geom(i_g)

    @ti.kernel
    def _kernel_get_free_verts(self, tensor: ti.types.ndarray()):
        for i, j, b in ti.ndrange(self.n_verts, 3, self._solver._B):
            idx_vert = i + self._verts_state_start
            tensor[b, i, j] = self._solver.free_verts_state.pos[idx_vert, b][j]

    @ti.kernel
    def _kernel_get_fixed_verts(self, tensor: ti.types.ndarray()):
        for i, j in ti.ndrange(self.n_verts, 3):
            idx_vert = i + self._verts_state_start
            tensor[i, j] = self._solver.fixed_verts_state.pos[idx_vert][j]

    @ezsim.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the entity (using collision geoms).

        Returns
        -------
        AABB : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The axis-aligned bounding box (AABB) of the entity (using collision geoms).
        """
        if self.n_geoms == 0:
            ezsim.raise_exception("Entity has no geoms.")

        verts = self.get_verts()
        AABB = torch.concatenate(
            [verts.min(axis=-2, keepdim=True)[0], verts.max(axis=-2, keepdim=True)[0]],
            axis=-2,
        )
        return AABB

    def _get_idx(self, idx_local, idx_local_max, idx_global_start=0, *, unsafe=False):
        # Handling default argument and special cases
        if idx_local is None:
            if unsafe:
                idx_global = slice(idx_global_start, idx_local_max + idx_global_start)
            else:
                idx_global = range(idx_global_start, idx_local_max + idx_global_start)
        elif isinstance(idx_local, (range, slice)):
            idx_global = range(
                (idx_local.start or 0) + idx_global_start,
                (idx_local.stop if idx_local.stop is not None else idx_local_max) + idx_global_start,
                idx_local.step or 1,
            )
        elif isinstance(idx_local, (int, np.integer)):
            idx_global = idx_local + idx_global_start
        elif isinstance(idx_local, (list, tuple)):
            try:
                idx_global = [i + idx_global_start for i in idx_local]
            except TypeError:
                ezsim.raise_exception("Expecting a sequence of integers for `idx_local`.")
        else:
            # Increment may be slow when dealing with heterogenuous data, so it must be avoided if possible
            if idx_global_start > 0:
                idx_global = idx_local + idx_global_start
            else:
                idx_global = idx_local

        # Early return if unsafe
        if unsafe:
            return idx_global

        # Perform a bunch of sanity checks
        _idx_global = torch.as_tensor(idx_global, dtype=ezsim.tc_int, device=ezsim.device).contiguous()
        if _idx_global is not idx_global:
            ezsim.logger.debug(ALLOCATE_TENSOR_WARNING)
        idx_global = torch.atleast_1d(_idx_global)

        if idx_global.ndim != 1:
            ezsim.raise_exception("Expecting a 1D tensor for `idx_local`.")
        if (idx_global < 0).any() or (idx_global >= idx_global_start + idx_local_max).any():
            ezsim.raise_exception("`idx_local` exceeds valid range.")

        return idx_global

    @ezsim.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, unsafe=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        """
        qs_idx = self._get_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        self._solver.set_qpos(qpos, qs_idx, envs_idx, unsafe=unsafe, skip_forward=zero_velocity)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the entity's dofs' positional gains for the PD controller.

        Parameters
        ----------
        kp : array_like
            The positional gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kp(kp, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the entity's dofs' velocity gains for the PD controller.

        Parameters
        ----------
        kv : array_like
            The velocity gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kv(kv, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the entity's dofs' force range.

        Parameters
        ----------
        lower : array_like
            The lower bounds of the force range.
        upper : array_like
            The upper bounds of the force range.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_range(lower, upper, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_stiffness(stiffness, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_invweight(self, invweight, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_invweight(invweight, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_armature(armature, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_damping(damping, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_velocity(self, velocity=None, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : array_like | None
            The velocity to set. Zero if not specified.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=False, unsafe=unsafe)

    @ezsim.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=True, unsafe=False):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_position(position, dofs_idx, envs_idx, unsafe=unsafe, skip_forward=zero_velocity)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Control the entity's dofs' motor force. This is used for force/torque control.

        Parameters
        ----------
        force : array_like
            The force to apply.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_force(force, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def control_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the PD controller's target velocity for the entity's dofs. This is used for velocity control.

        Parameters
        ----------
        velocity : array_like
            The target velocity to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_velocity(velocity, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the PD controller's target position for the entity's dofs. This is used for position control.

        Parameters
        ----------
        position : array_like
            The target position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position(position, dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the entity's qpos.

        Parameters
        ----------
        qs_idx_local : None | array_like, optional
            The indices of the qpos to get. If None, all qpos will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        qpos : torch.Tensor, shape (n_qs,) or (n_envs, n_qs)
            The entity's qpos.
        """
        qs_idx = self._get_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        return self._solver.get_qpos(qs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the entity's dofs' internal control force, computed based on the position/velocity control command.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        control_force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' internal control force.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_control_force(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the entity's dofs' internal force at the current time step.

        Note
        ----
        Different from `get_dofs_control_force`, this function returns the actual internal force experienced by all the dofs at the current time step.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' force.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_velocity(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_position(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the positional gain (kp) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The positional gain (kp) for the entity's dofs.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kp(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the velocity gain (kv) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kv : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The velocity gain (kv) for the entity's dofs.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kv(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force_range(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Get the positional limits (min and max) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the positional limits for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the positional limits for the entity's dofs.
        """
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_limit(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_stiffness(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_invweight(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_armature(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None, *, unsafe=False):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_damping(dofs_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def get_mass_mat(self, envs_idx=None, decompose=False, *, unsafe=False):
        dofs_idx = self._get_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_mass_mat(dofs_idx, envs_idx, decompose, unsafe=unsafe)

    @ezsim.assert_built
    def zero_all_dofs_velocity(self, envs_idx=None, *, unsafe=False):
        """
        Zero the velocity of all the entity's dofs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx_local = torch.arange(self.n_dofs, dtype=ezsim.tc_int, device=ezsim.device)
        self.set_dofs_velocity(None, dofs_idx_local, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def detect_collision(self, env_idx=0):
        """
        Detects collision for the entity. This only supports a single environment.

        Note
        ----
        This function re-detects real-time collision for the entity, so it doesn't rely on scene.step() and can be used for applications like motion planning, which doesn't require physical simulation during state sampling.

        Parameters
        ----------
        env_idx : int, optional
            The index of the environment. Defaults to 0.
        """

        all_collision_pairs = self._solver.detect_collision(env_idx)
        collision_pairs = all_collision_pairs[
            np.logical_and(
                all_collision_pairs >= self.geom_start,
                all_collision_pairs < self.geom_end,
            ).any(axis=1)
        ]
        return collision_pairs

    @ezsim.assert_built
    def get_contacts(self, with_entity=None, exclude_self_contact=False):
        """
        Returns contact information computed during the most recent `scene.step()`.
        If `with_entity` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When `with_entity` is `self`, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : RigidEntity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        contacts_info = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)

        logical_operation = torch.logical_xor if exclude_self_contact else torch.logical_or
        if with_entity is not None and self.idx == with_entity.idx:
            if exclude_self_contact:
                ezsim.raise_exception("`with_entity` is self but `exclude_self_contact` is True.")
            logical_operation = torch.logical_and

        valid_mask = logical_operation(
            torch.logical_and(
                contacts_info["geom_a"] >= self.geom_start,
                contacts_info["geom_a"] < self.geom_end,
            ),
            torch.logical_and(
                contacts_info["geom_b"] >= self.geom_start,
                contacts_info["geom_b"] < self.geom_end,
            ),
        )
        if with_entity is not None and self.idx != with_entity.idx:
            valid_mask = torch.logical_and(
                valid_mask,
                torch.logical_or(
                    torch.logical_and(
                        contacts_info["geom_a"] >= with_entity.geom_start,
                        contacts_info["geom_a"] < with_entity.geom_end,
                    ),
                    torch.logical_and(
                        contacts_info["geom_b"] >= with_entity.geom_start,
                        contacts_info["geom_b"] < with_entity.geom_end,
                    ),
                ),
            )

        if self._solver.n_envs == 0:
            contacts_info = {key: value[valid_mask] for key, value in contacts_info.items()}
        else:
            contacts_info["valid_mask"] = valid_mask

        contacts_info["force_a"] = -contacts_info["force"]
        contacts_info["force_b"] = +contacts_info["force"]
        del contacts_info["force"]

        return contacts_info

    def get_links_net_contact_force(self, envs_idx=None, *, unsafe=False):
        """
        Returns net force applied on each links due to direct external contacts.

        Returns
        -------
        entity_links_force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net force applied on each links due to direct external contacts.
        """
        tensor = ti_field_to_torch(
            self._solver.links_state.contact_force, envs_idx, slice(self.link_start, self.link_end), transpose=True
        )
        return tensor.squeeze(0) if self._solver.n_envs == 0 else tensor

    def set_friction_ratio(self, friction_ratio, links_idx_local=None, envs_idx=None,*,unsafe=False):
        """
        Set the friction ratio of the geoms of the specified links.

        Parameters
        ----------
        friction_ratio : torch.Tensor, shape (n_envs, n_links)
            The friction ratio
        links_idx_local : array_like
            The indices of the links to set friction ratio.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx_local = self._get_idx(links_idx_local, self.n_links, 0, unsafe=unsafe)
        links_n_geoms = torch.tensor(
            [self._links[i_l].n_geoms for i_l in links_idx_local], dtype=ezsim.tc_int, device=ezsim.device
        )
        links_friction_ratio = torch.as_tensor(friction_ratio, dtype=ezsim.tc_float, device=ezsim.device)
        geoms_friction_ratio = torch.repeat_interleave(links_friction_ratio, links_n_geoms, dim=-1)
        geoms_idx = torch.tensor(
            [i_g for i_l in links_idx_local for i_g in range(self._links[i_l].geom_start, self._links[i_l].geom_end)],
            dtype=ezsim.tc_int,
            device=ezsim.device,
        )
        self._solver.set_geoms_friction_ratio(geoms_friction_ratio, geoms_idx, envs_idx, unsafe=unsafe)

    def set_friction(self, friction):
        """
        Set the friction coefficient of all the links (and in turn, geometries) of the rigid entity.
        
        Note
        ----
        The friction coefficient associated with a pair of geometries in contact is defined as the maximum between
        their respective values, so one must be careful the set the friction coefficient properly for both of them.

        Warning
        -------
        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.
        
        Parameters
        ----------
        friction : float
            The friction coefficient to set.
        """

        if friction < 1e-2 or friction > 5.0:
            ezsim.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        for link in self._links:
            link.set_friction(friction)

    def set_mass_shift(self, mass_shift, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the mass shift of specified links.
        Parameters
        ----------
        mass : torch.Tensor, shape (n_envs, n_links)
            The mass shift
        links_idx_local : array_like
            The indices of the links to set mass shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_mass_shift(mass_shift, links_idx, envs_idx, unsafe=unsafe)

    def set_COM_shift(self, com_shift, links_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the center of mass (COM) shift of specified links.
        Parameters
        ----------
        com : torch.Tensor, shape (n_envs, n_links, 3)
            The COM shift
        links_idx_local : array_like
            The indices of the links to set COM shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_COM_shift(com_shift, links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_links_inertial_mass(self, inertial_mass, links_idx_local=None, envs_idx=None, *, unsafe=False):
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_inertial_mass(inertial_mass, links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_links_invweight(self, invweight, links_idx_local=None, envs_idx=None, *, unsafe=False):
        links_idx = self._get_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_invweight(invweight, links_idx, envs_idx, unsafe=unsafe)

    @ezsim.assert_built
    def set_mass(self, mass):
        """
        Set the mass of the entity.

        Parameters
        ----------
        mass : float
            The mass to set.
        """
        ratio = float(mass) / self.get_mass()
        for link in self.links:
        #     original_mass_distribution.append(link.get_mass())
        # original_mass_distribution = np.array(original_mass_distribution)
        # original_mass_distribution /= np.sum(original_mass_distribution)
        # for link, mass_ratio in zip(self.links, original_mass_distribution):
        #     link.set_mass(mass * mass_ratio)
            link.set_mass(link.get_mass() * ratio)

    @ezsim.assert_built
    def get_mass(self):
        """
        Get the total mass of the entity in kg.

        Returns
        -------
        mass : float
            The total mass of the entity in kg.
        """
        mass = 0.0
        for link in self.links:
            mass += link.get_mass()
        return mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

    @property
    def init_qpos(self):
        """The initial qpos of the entity."""
        if self.joints:
            return np.concatenate([joint.init_qpos for joint in self.joints])
        return np.array([])

    @property
    def n_qs(self):
        """The number of `q` (generalized coordinates) of the entity."""
        if self._is_built:
            return self._n_qs
        return sum(joint.n_qs for joint in self.joints)

    @property
    def n_links(self):
        """The number of `RigidLink` in the entity."""
        return len(self._links)

    @property
    def n_joints(self):
        """The number of `RigidJoint` in the entity."""
        return sum(map(len, self._joints))

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        if self._is_built:
            return self._n_dofs
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def n_geoms(self):
        """The number of `RigidGeom` in the entity."""
        return sum(link.n_geoms for link in self._links)

    @property
    def n_cells(self):
        """The number of sdf cells in the entity."""
        return sum(link.n_cells for link in self._links)

    @property
    def n_verts(self):
        """The number of vertices (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_verts for link in self._links)

    @property
    def n_faces(self):
        """The number of faces (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_faces for link in self._links)

    @property
    def n_edges(self):
        """The number of edges (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_edges for link in self._links)

    @property
    def n_vgeoms(self):
        """The number of vgeoms (visual geoms - `RigidVisGeom`) in the entity."""
        return sum(link.n_vgeoms for link in self._links)

    @property
    def n_vverts(self):
        """The number of vverts (visual vertices, from vgeoms) in the entity."""
        return sum([link.n_vverts for link in self._links])

    @property
    def n_vfaces(self):
        """The number of vfaces (visual faces, from vgeoms) in the entity."""
        return sum([link.n_vfaces for link in self._links])

    @property
    def geom_start(self):
        """The index of the entity's first RigidGeom in the scene."""
        return self._geom_start

    @property
    def geom_end(self):
        """The index of the entity's last RigidGeom in the scene *plus one*."""
        return self._geom_start + self.n_geoms

    @property
    def cell_start(self):
        """The start index the entity's sdf cells in the scene."""
        return self._cell_start

    @property
    def cell_end(self):
        """The end index the entity's sdf cells in the scene *plus one*."""
        return self._cell_start + self.n_cells

    @property
    def base_link_idx(self):
        """The index of the entity's base link in the scene."""
        return self._link_start

    @property
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

    @property
    def link_start(self):
        """The index of the entity's first RigidLink in the scene."""
        return self._link_start

    @property
    def link_end(self):
        """The index of the entity's last RigidLink in the scene *plus one*."""
        return self._link_start + self.n_links

    @property
    def joint_start(self):
        """The index of the entity's first RigidJoint in the scene."""
        return self._joint_start

    @property
    def joint_end(self):
        """The index of the entity's last RigidJoint in the scene *plus one*."""
        return self._joint_start + self.n_joints

    @property
    def dof_start(self):
        """The index of the entity's first degree of freedom (DOF) in the scene."""
        return self._dof_start

    @property
    def dof_end(self):
        """The index of the entity's last degree of freedom (DOF) in the scene *plus one*."""
        return self._dof_start + self.n_dofs

    @property
    def vert_start(self):
        """The index of the entity's first `vert` (collision vertex) in the scene."""
        return self._vert_start

    @property
    def vvert_start(self):
        """The index of the entity's first `vvert` (visual vertex) in the scene."""
        return self._vvert_start

    @property
    def face_start(self):
        """The index of the entity's first `face` (collision face) in the scene."""
        return self._face_start

    @property
    def vface_start(self):
        """The index of the entity's first `vface` (visual face) in the scene."""
        return self._vface_start

    @property
    def edge_start(self):
        """The index of the entity's first `edge` (collision edge) in the scene."""
        return self._edge_start

    @property
    def q_start(self):
        """The index of the entity's first `q` (generalized coordinates) in the scene."""
        return self._q_start

    @property
    def q_end(self):
        """The index of the entity's last `q` (generalized coordinates) in the scene *plus one*."""
        return self._q_start + self.n_qs

    @property
    def geoms(self) -> list[RigidGeom]:
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        else:
            geoms = ezsim.List()
            for link in self._links:
                geoms += link.geoms
            return geoms

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        else:
            vgeoms = ezsim.List()
            for link in self._links:
                vgeoms += link.vgeoms
            return vgeoms

    @property
    def links(self) -> list[RigidLink]:
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self):
        """The list of joints (`RigidJoint`) in the entity."""
        return tuple(chain.from_iterable(self._joints))

    @property
    def joints_by_links(self):
        """The list of joints (`RigidJoint`) in the entity grouped by parent links."""
        return self._joints

    @property
    def base_link(self):
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self):
        """The base joint of the entity"""
        return self._joints[0][0]

    @property
    def n_equalities(self):
        """The number of equality constraints in the entity."""
        return len(self._equalities)

    @property
    def equality_start(self):
        """The index of the entity's first RigidEquality in the scene."""
        return self._equality_start

    @property
    def equality_end(self):
        """The index of the entity's last RigidEquality in the scene *plus one*."""
        return self._equality_start + self.n_equalities

    @property
    def equalities(self):
        """The list of equality constraints (`RigidEquality`) in the entity."""
        return self._equalities

    @property
    def is_free(self):
        """Whether the entity is free to move."""
        return self._is_free
