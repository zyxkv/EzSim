
from ezsim.engine.entities.rigid_entity.rigid_entity import RigidEntity
from ezsim.engine.entities.rigid_entity.rigid_geom import RigidGeom

from .ray import Plane, Ray, RayHit
from .vec3 import Pose, Quat, Vec3, Color

from ezsim.engine.entities.rigid_entity.rigid_link import RigidLink

MOUSE_SPRING_POSITION_CORRECTION_FACTOR = 1.0
MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR = 1.0

def _ensure_torch_imported() -> None:
    global torch
    import torch

class MouseSpring:
    def __init__(self) -> None:
        # NOTE: change mouse-spring attach to hel_link instead of held_geom.
        self.held_link: RigidLink | None = None
        # self.held_geom: RigidGeom | None = None
        self.held_point_in_local: Vec3 | None = None
        self.prev_control_point: Vec3 | None = None

    def attach(self, picked_link: RigidLink, control_point: Vec3) -> None:
        self.held_link = picked_link
        pose: Pose = Pose.from_geom(self.held_link)
        self.held_point_in_local = pose.inverse_transform_point(control_point)
        self.prev_control_point = control_point

    def detach(self) -> None:
        self.held_link = None

    def apply_force(self, control_point: Vec3, delta_time: float):
        _ensure_torch_imported()
        # note when threaded: apply_force is called before attach!
        # note2: that was before we added a lock to ViewerInteraction; this might be fixed now
        if not self.held_link:
            return
        
        # works ok:
        # delta: Vec3 = control_point - self.prev_control_point
        # pos = Vec3.from_tensor(self.held_geom.entity.get_pos())
        # pos = pos + delta
        # self.held_geom.entity.set_pos(pos.as_tensor())
        self.prev_control_point = control_point

        # do simple force on COM only:
        link: RigidLink = self.held_link
        link_pos: Vec3 = Vec3.from_tensor(link.get_pos())
        lin_vel: Vec3 = Vec3.from_tensor(link.get_vel())
        ang_vel: Vec3 = Vec3.from_tensor(link.get_ang())
        link_pose: Pose = Pose.from_link(link)
        held_point_in_world: Vec3 = link_pose.transform_point(self.held_point_in_local)
        # note: we should assert earlier that link inertial_pos/quat are not None
        # todo: verify inertial_pos/quat are stored in local frame
        link_T_principal: Pose = Pose(Vec3.from_arraylike(link.inertial_pos), Quat.from_arraylike(link.inertial_quat))
        world_T_principal: Pose = link_pose * link_T_principal

        arm_in_principal: Vec3 = link_T_principal.inverse_transform_point(self.held_point_in_local)   # for non-spherical inertia
        arm_in_world: Vec3 = world_T_principal.rot * arm_in_principal  # for spherical inertia


        pos_err_v: Vec3 = control_point - held_point_in_world
        # vel_err_v: Vec3 = Vec3.zero() - lin_vel
        inv_mass: float = float(1.0 / link.get_mass() if link.get_mass() > 0.0 else 0.0)
        #NOTE: inv_spherical_inertia = 0.0 would be a body that's fixed rotationally, 
        # or is infinitely inertive and torques don't affect its ang_vel
        inv_spherical_inertia: float = float(1.0 / link.inertial_i[0, 0] if link.inertial_i[0, 0] > 0.0 else 0.0)

        inv_dt: float = 1.0 / delta_time
        # these are temporary values, till we fix an issue with apply_links_external_force.
        # after fixing it, use tau = damp = 1.0:
        tau: float = MOUSE_SPRING_POSITION_CORRECTION_FACTOR
        damp: float = MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR

        total_impulse: Vec3 = Vec3.zero()
        total_torque_impulse: Vec3 = Vec3.zero()

        for i in range(3*4):
            body_point_vel: Vec3 = lin_vel + ang_vel.cross(arm_in_world)
            vel_err_v: Vec3 = Vec3.zero() - body_point_vel
            dir: Vec3 = Vec3.zero()
            dir.v[i % 3] = 1.0
            pos_err: float = dir.dot(pos_err_v)
            vel_err: float = dir.dot(vel_err_v)
            error: float = tau * pos_err * inv_dt + damp * vel_err
            arm_x_dir: Vec3 = arm_in_world.cross(dir)
            virtual_mass: float = 1.0 / (inv_mass + arm_x_dir.sqr_magnitude() * inv_spherical_inertia + 1e-24)
            lin_vel += impulse * inv_mass * dir
            ang_vel += impulse * inv_spherical_inertia * arm_x_dir
            total_impulse.v[i % 3] += impulse
            total_torque_impulse += impulse * arm_x_dir

        # for i in range(3):
        #     dir: Vec3 = Vec3.zero()
        #     dir.v[i] = 1.0
        #     pos_err: float = dir.dot(pos_err_v)
        #     vel_err: float = dir.dot(vel_err_v)
        #     error: float = tau * pos_err * inv_dt + damp * vel_err
        #     virtual_mass: float = 1.0 / (inv_mass + 1e-24)
        #     impulse: float = error * virtual_mass

        #     lin_vel += impulse * dir * inv_mass
        #     total_impulse.v[i] = impulse

        # Apply the new force
        total_force = total_impulse * inv_dt
        total_torque = total_torque_impulse * inv_dt
        force_tensor: torch.Tensor = total_force.as_tensor().unsqueeze(0)
        torque_tensor: torch.Tensor = total_torque.as_tensor().unsqueeze(0)
        link.solver.apply_links_external_force(force_tensor, (link.idx,), ref='link_com', local=False)
        link.solver.apply_links_external_torque(torque_tensor, (link.idx,), ref='link_com', local=False)
    @property
    def is_attached(self) -> bool:
        return self.held_link is not None
