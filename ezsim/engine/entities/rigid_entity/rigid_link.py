from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
import taichi as ti
import torch

import ezsim
import trimesh
from ezsim.repr_base import RBC
from ezsim.utils import geom as gu

from .rigid_geom import RigidGeom, RigidVisGeom

if TYPE_CHECKING:
    from .rigid_entity import RigidEntity
    from ezsim.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


@ti.data_oriented
class RigidLink(RBC):
    """
    RigidLink class. One RigidEntity consists of multiple RigidLinks, each of which is a rigid body and could consist of multiple RigidGeoms (`link.geoms`, for collision) and RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity,
        name,
        idx,
        joint_start,
        n_joints,
        geom_start,
        cell_start,
        vert_start,
        face_start,
        edge_start,
        verts_state_start,
        vgeom_start,
        vvert_start,
        vface_start,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass,
        parent_idx,
        root_idx,
        invweight,
        visualize_contact,
    ):
        self._name: str = name
        self._entity: "RigidEntity" = entity
        self._solver: "RigidSolver" = entity.solver
        self._entity_idx_in_solver = entity.idx

        self._uid = ezsim.UID()
        self._idx: int = idx
        self._parent_idx: int = parent_idx  # -1 if no parent
        self._root_idx: int | None = root_idx  # None if no root
        self._child_idxs: list[int] = list()
        self._invweight: float | None = invweight

        self._joint_start: int = joint_start
        self._n_joints: int = n_joints

        self._geom_start: int = geom_start
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start
        self._vgeom_start: int = vgeom_start
        self._vvert_start: int = vvert_start
        self._vface_start: int = vface_start

        # Link position & rotation at creation time:
        self._pos: ArrayLike = pos
        self._quat: ArrayLike = quat
        # Link's center-of-mass position & principal axes frame rotation at creation time:
        if inertial_pos is not None:
            inertial_pos = np.asarray(inertial_pos, dtype=ezsim.np_float)
        self._inertial_pos: ArrayLike | None = inertial_pos
        if inertial_quat is not None:
            inertial_quat = np.asarray(inertial_quat, dtype=ezsim.np_float)
        self._inertial_quat: ArrayLike | None = inertial_quat
        self._inertial_mass = inertial_mass
        self._inertial_i = inertial_i

        self._visualize_contact = visualize_contact

        self._geoms: list[RigidGeom] = ezsim.List()
        self._vgeoms = ezsim.List()

    def _build(self):
        for geom in self._geoms:
            geom._build()

        for vgeom in self._vgeoms:
            vgeom._build()

        self._init_mesh = self._compose_init_mesh()

        # find root link and check if link is fixed
        solver_links = self._solver.links
        link = self
        is_fixed = all(joint.type is ezsim.JOINT_TYPE.FIXED for joint in self.joints)
        while link.parent_idx > -1:
            link = solver_links[link.parent_idx]
            if not all(joint.type is ezsim.JOINT_TYPE.FIXED for joint in link.joints):
                is_fixed = False
        if self._root_idx is None:
            self._root_idx = ezsim.np_int(link.idx)
        # self.is_fixed: np.int32 = ezsim.np_int(is_fixed)  # note: type inconsistent with is_built, is_free, is_leaf: bool
        self.is_fixed = is_fixed
        
        # inertial_mass and inertia_i
        if self._inertial_mass is None:
            if len(self._geoms) == 0 and len(self._vgeoms) == 0:
                self._inertial_mass = 0.0
            else:
                if self._init_mesh.is_watertight:
                    self._inertial_mass = self._init_mesh.volume * self.entity.material.rho
                else:  # TODO: handle non-watertight mesh
                    self._inertial_mass = 1.0

        # Postpone computation of inverse weight if not specified
        if self._invweight is None:
            self._invweight = np.full((2,), fill_value=-1.0, dtype=ezsim.np_float)

        # inertial_pos
        if self._inertial_pos is None:
            if self._init_mesh is None:
                self._inertial_pos = gu.zero_pos()
            else:
                self._inertial_pos = np.array(self._init_mesh.center_mass, dtype=ezsim.np_float)

        # inertial_i
        if self._inertial_i is None:
            # FIXME: Why coef 0.4 ???
            if self._init_mesh is None:  # use sphere inertia with radius 0.1
                self._inertial_i = 0.4 * self._inertial_mass * 0.1**2 * np.eye(3)

            else:
                # attempt to fix non-watertight mesh by convexifying
                inertia_mesh = self._init_mesh.copy()
                if not inertia_mesh.is_watertight:
                    inertia_mesh = trimesh.convex.convex_hull(inertia_mesh)

                if inertia_mesh.is_watertight and self._init_mesh.mass > 0:
                    # TODO: check if this is correct. This is correct if the inertia frame is w.r.t to link frame
                    T_inertia = gu.trans_quat_to_T(self._inertial_pos, self._inertial_quat)
                    self._inertial_i = (
                        self._init_mesh.moment_inertia_frame(T_inertia) / self._init_mesh.mass * self._inertial_mass
                    )

                else:  # approximate with a sphere
                    self._inertial_i = (
                        0.4
                        * self._inertial_mass
                        * (max(self._init_mesh.bounds[1] - self._init_mesh.bounds[0]) / 2.0) ** 2
                        * np.eye(3)
                    )

        self._inertial_i = np.asarray(self._inertial_i, dtype=ezsim.np_float)

        # override invweight if fixed
        if is_fixed:
            self._invweight = np.zeros((2,), dtype=ezsim.np_float)

        import ezsim.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp

        self.rsd = rigid_solver_decomp

    def _compose_init_mesh(self):
        if len(self._geoms) == 0 and len(self._vgeoms) == 0:
            return None
        else:
            init_verts = []
            init_faces = []
            vert_offset = 0
            if len(self._geoms) > 0:
                for geom in self._geoms:
                    init_verts.append(gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat))
                    init_faces.append(geom.init_faces + vert_offset)
                    vert_offset += geom.init_verts.shape[0]
            elif len(self._vgeoms) > 0:  # use vgeom if there's no geom
                for vgeom in self._vgeoms:
                    init_verts.append(gu.transform_by_trans_quat(vgeom.init_vverts, vgeom.init_pos, vgeom.init_quat))
                    init_faces.append(vgeom.init_vfaces + vert_offset)
                    vert_offset += vgeom.init_vverts.shape[0]
            init_verts = np.concatenate(init_verts)
            init_faces = np.concatenate(init_faces)
            return trimesh.Trimesh(init_verts, init_faces)

    def _add_geom(
        self,
        mesh,
        init_pos,
        init_quat,
        type,
        friction,
        sol_params,
        center_init=None,
        needs_coup=False,
        contype=1,
        conaffinity=1,
        data=None,
    ):
        geom = RigidGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            mesh=mesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            friction=friction,
            sol_params=sol_params,
            center_init=center_init,
            needs_coup=needs_coup,
            contype=contype,
            conaffinity=conaffinity,
            data=data,
        )
        self._geoms.append(geom)

    def _add_vgeom(self, vmesh, init_pos, init_quat):
        vgeom = RigidVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            vmesh=vmesh,
            init_pos=init_pos,
            init_quat=init_quat,
        )
        self._vgeoms.append(vgeom)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @ezsim.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the position. If None, get the position of all environments. Default is None.
        """
        return self._solver.get_links_pos([self._idx], envs_idx).squeeze(-2)

    @ezsim.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the quaternion. If None, get the quaternion of all environments. Default is None.
        """
        return self._solver.get_links_quat([self._idx], envs_idx).squeeze(-2)

    @ezsim.assert_built
    def get_vel(self, envs_idx=None) -> torch.Tensor:
        """
        Get the linear velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the linear velocity. If None, get the linear velocity of all environments. Default is None.
        """
        return self._solver.get_links_vel([self._idx], envs_idx).squeeze(-2)

    @ezsim.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """
        Get the angular velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the angular velocity. If None, get the angular velocity of all environments. Default is None.
        """
        return self._solver.get_links_ang([self._idx], envs_idx).squeeze(-2)

    @ezsim.assert_built
    def get_verts(self):
        """
        Get the vertices of the link's collision body (concatenation of all `link.geoms`) in the world frame.
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
    def get_vverts(self):
        """
        Get the vertices of the link's visualization body (concatenation of all `link.vgeoms`) in the world frame.
        """
        tensor = torch.empty(self._solver._batch_shape((self.n_vverts, 3), True), dtype=ezsim.tc_float, device=ezsim.device)
        self._kernel_get_vverts(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_vverts(self, tensor: ti.types.ndarray()):
        for i_vg_, i_b in ti.ndrange(self.n_vgeoms, self._solver._B):
            i_vg = i_vg_ + self._vgeom_start
            for i_v in range(self._solver.vgeoms_info.vvert_start[i_vg], self._solver.vgeoms_info.vvert_end[i_vg]):
                vvert_pos = gu.ti_transform_by_trans_quat(
                    self._solver.vverts_info.init_pos[i_v],
                    self._solver.vgeoms_state.pos[i_vg, i_b],
                    self._solver.vgeoms_state.quat[i_vg, i_b],
                )
                for j in range(3):
                    tensor[i_b, i_v - self._vvert_start, j] = vvert_pos[j]

    @ezsim.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the link's collision body (concatenation of all `link.geoms`) in the world frame.
        """
        verts = self.get_verts()
        AABB = torch.concatenate(
            [verts.min(axis=-2, keepdim=True)[0], verts.max(axis=-2, keepdim=True)[0]],
            axis=-2,
        )
        return AABB

    @ezsim.assert_built
    def get_vAABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the link's visual body (concatenation of all `link.vgeoms`) in the world frame.
        """
        vverts = self.get_vverts()
        AABB = torch.concatenate(
            [vverts.min(axis=-2, keepdim=True)[0], vverts.max(axis=-2, keepdim=True)[0]],
            axis=-2,
        )
        return AABB

    @ezsim.assert_built
    def set_mass(self, mass):
        """
        Set the mass of the link.
        """
        if self.is_fixed:
            ezsim.warning(f"Updating the mass of a link that is fixed wrt world has no effect, skipping.")
            return 
        
        if mass < ezsim.EPS:
            ezsim.raise_exception(f"Attempt to set mass of link '{self.name}' to {mass}. Mass must be strictly positive.")

        ratio = float(mass) / self._inertial_mass
        self._inertial_mass *= ratio
        if self._invweight is not None:
            self._invweight /= ratio
        self._inertial_i *= ratio

        self.rsd.kernel_adjust_link_inertia(
            link_idx=self.idx,
            ratio=ratio,
            links_info=self._solver.links_info,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
        )

    @ezsim.assert_built
    def get_mass(self):
        """
        Get the mass of the link.
        """
        return self._inertial_mass

    def set_friction(self, friction):
        """
        Set the friction of all the link's geoms.
        """
        for geom in self._geoms:
            geom.set_friction(friction)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        The unique ID of the link.
        """
        return self._uid

    @property
    def name(self):
        """
        The name of the link.
        """
        return self._name

    @property
    def entity(self):
        """
        The entity that the link belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        The solver that the link belongs to.
        """
        return self._solver

    @property
    def visualize_contact(self):
        """
        Whether to visualize the contact of the link.
        """
        return self._visualize_contact

    @property
    def joints(self):
        """
        The sequence of joints that connects the link to its parent link.
        """
        return self._solver.joints[self.joint_start : self.joint_end]

    @property
    def n_joints(self):
        """
        Number of the joints that connects the link to its parent link.
        """
        return self._n_joints

    @property
    def joint_start(self):
        """
        The start index of the link's joints in the RigidSolver.
        """
        return self._joint_start

    @property
    def joint_end(self):
        """
        The end index of the link's joints in the RigidSolver.
        """
        return self._joint_start + self.n_joints

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def dof_start(self):
        """The index of the link's first degree of freedom (DOF) in the scene."""
        if len(self.joints) == 0:
            return -1
        return self.joints[0].dof_start

    @property
    def dof_end(self):
        """The index of the link's last degree of freedom (DOF) in the scene *plus one*."""
        if len(self.joints) == 0:
            return -1
        return self.joints[-1].dof_end

    @property
    def n_qs(self):
        """Returns the number of `q` variables of the link."""
        return sum(joint.n_qs for joint in self.joints)

    @property
    def q_start(self):
        """Returns the starting index of the `q` variables of the link in the rigid solver."""
        if len(self.joints) == 0:
            return -1
        return self.joints[0].q_start

    @property
    def q_end(self):
        """Returns the last index of the `q` variables of the link in the rigid solver *plus one*."""
        if len(self.joints) == 0:
            return -1
        return self.joints[-1].q_end

    @property
    def idx(self):
        """
        The global index of the link in the RigidSolver.
        """
        return self._idx

    @property
    def parent_idx(self):
        """
        The global index of the link's parent link in the RigidSolver. If the link is the root link, return -1.
        """
        return self._parent_idx

    @property
    def root_idx(self):
        """
        The global index of the link's root link in the RigidSolver.
        """
        return self._root_idx

    @property
    def child_idxs(self):
        """
        The global indices of the link's child links in the RigidSolver.
        """
        return self._child_idxs

    @property
    def idx_local(self):
        """
        The local index of the link in the entity.
        """
        return self._idx - self._entity.link_start

    @property
    def parent_idx_local(self):
        """
        The local index of the link's parent link in the entity. If the link is the root link, return -1.
        """
        # TODO: check for parent links outside of the current entity (caused by scene.link_entities())
        if self._parent_idx >= 0:
            return self._parent_idx - self._entity.link_start
        return self._parent_idx

    @property
    def child_idxs_local(self):
        """
        The local indices of the link's child links in the entity.
        """
        # TODO: check for child links outside of the current entity (caused by scene.link_entities())
        return [idx - self._entity.link_start if idx >= 0 else idx for idx in self._child_idxs]

    @property
    def is_leaf(self):
        """
        Whether the link is a leaf link (i.e., has no child links).
        """
        return len(self._child_idxs) == 0

    @property
    def invweight(self):
        """
        The invweight of the link.
        """
        if self._invweight is None:
            self._invweight = self._solver.get_links_invweight([self._idx]).cpu().numpy()[..., 0, :]
        return self._invweight

    @property
    def pos(self):
        """
        The initial position of the link. For real-time position, use `link.get_pos()`.
        """
        return self._pos

    @property
    def quat(self):
        """
        The initial quaternion of the link. For real-time quaternion, use `link.get_quat()`.
        """
        return self._quat

    @property
    def inertial_pos(self):
        """
        The initial position of the link's inertial frame.
        """
        return self._inertial_pos

    @property
    def inertial_quat(self):
        """
        The initial quaternion of the link's inertial frame.
        """
        return self._inertial_quat

    @property
    def inertial_mass(self):
        """
        The initial mass of the link.
        """
        return self._inertial_mass

    @property
    def inertial_i(self):
        """
        The inerial matrix of the link.
        """
        return self._inertial_i

    @property
    def geoms(self):
        """
        The list of the link's collision geometries (`RigidGeom`).
        """
        return self._geoms

    @property
    def vgeoms(self):
        """
        The list of the link's visualization geometries (`RigidVisGeom`).
        """
        return self._vgeoms

    @property
    def n_geoms(self):
        """
        Number of the link's collision geometries.
        """
        return len(self._geoms)

    @property
    def geom_start(self):
        """
        The start index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start

    @property
    def geom_end(self):
        """
        The end index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start + self.n_geoms

    @property
    def n_vgeoms(self):
        """
        Number of the link's visualization geometries (`vgeom`).
        """
        return len(self._vgeoms)

    @property
    def vgeom_start(self):
        """
        The start index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start

    @property
    def vgeom_end(self):
        """
        The end index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start + self.n_vgeoms

    @property
    def n_cells(self):
        """
        Number of sdf cells of all the link's geoms.
        """
        return sum([geom.n_cells for geom in self._geoms])

    @property
    def n_verts(self):
        """
        Number of vertices of all the link's geoms.
        """
        return sum([geom.n_verts for geom in self._geoms])

    @property
    def n_vverts(self):
        """
        Number of vertices of all the link's vgeoms.
        """
        return sum([vgeom.n_vverts for vgeom in self._vgeoms])

    @property
    def n_faces(self):
        """
        Number of faces of all the link's geoms.
        """
        return sum([geom.n_faces for geom in self._geoms])

    @property
    def n_vfaces(self):
        """
        Number of faces of all the link's vgeoms.
        """
        return sum([vgeom.n_vfaces for vgeom in self._vgeoms])

    @property
    def n_edges(self):
        """
        Number of edges of all the link's geoms.
        """
        return sum([geom.n_edges for geom in self._geoms])

    @property
    def is_built(self):
        """
        Whether the entity the link belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        """
        Whether the entity the link belongs to is free.
        """
        return self.entity.is_free

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self._repr_type())}: {self._uid}, name: '{self._name}', idx: {self._idx}"
