import os
import xml.etree.ElementTree as ET

import torch
import taichi as ti

import ezsim
import ezsim.utils.misc as mu

from .rigid_entity import RigidEntity


@ti.data_oriented
class DroneEntity(RigidEntity):
    def _load_scene(self, morph, surface):
        super()._load_scene(morph, surface)

        # additional drone specific attributes
        properties = ET.parse(os.path.join(mu.get_assets_dir(), morph.file)).getroot()[0].attrib
        self._KF = float(properties["kf"])
        self._KM = float(properties["km"])

        self._n_propellers = len(morph.propellers_link_name)

        propellers_link = ezsim.List([self.get_link(name) for name in morph.propellers_link_name])
        self._propellers_link_idxs = torch.tensor(
            [link.idx for link in propellers_link], dtype=ezsim.tc_int, device=ezsim.device
        )
        try:
            self._propellers_vgeom_idxs = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_link], dtype=ezsim.tc_int, device=ezsim.device
            )
            self._animate_propellers = True
        except Exception:
            ezsim.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=ezsim.tc_float, device=ezsim.device)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = torch.zeros(
            self._solver._batch_shape(self._n_propellers), dtype=ezsim.tc_float, device=ezsim.device
        )
        self._prev_prop_t = None

    def set_propellels_rpm(self, propellels_rpm):
        """
        Set the RPM (revolutions per minute) for each propeller in the drone.

        Parameters
        ----------
        propellels_rpm : array-like or torch.Tensor
            A tensor or array of shape (n_propellers,) or (n_envs, n_propellers) specifying
            the desired RPM values for each propeller. Must be non-negative.

        Raises
        ------
        RuntimeError
            If the method is called more than once per simulation step, or if the input shape
            does not match the number of propellers, or contains negative values.
        """
        if self._prev_prop_t == self.sim.cur_step_global:
            ezsim.raise_exception("`set_propellels_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        propellels_rpm = self.solver._process_dim(
            torch.as_tensor(propellels_rpm, dtype=ezsim.tc_float, device=ezsim.device)
        ).T.contiguous()
        if len(propellels_rpm) != len(self._propellers_link_idxs):
            ezsim.raise_exception("Last dimension of `propellels_rpm` does not match `entity.n_propellers`.")
        if torch.any(propellels_rpm < 0):
            ezsim.raise_exception("`propellels_rpm` cannot be negative.")
        self._propellers_revs = (self._propellers_revs + propellels_rpm) % (60 / self.solver.dt)

        self.solver._kernel_set_drone_rpm(
            self._n_propellers,
            self._propellers_link_idxs,
            propellels_rpm,
            self._propellers_spin,
            self.KF,
            self.KM,
            self._model == "RACE",
            self.solver.links_state,
        )

    def update_propeller_vgeoms(self):
        """
        Update the visual geometry of the propellers for animation based on their current rotation.

        This method is a no-op if animation is disabled due to missing visual geometry.
        """
        if self._animate_propellers:
            self.solver._update_drone_propeller_vgeoms(
                self._n_propellers, self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
            )

    @property
    def model(self):
        """The model type of the drone."""
        return self._model

    @property
    def KF(self):
        """The drone's thrust coefficient."""
        return self._KF

    @property
    def KM(self):
        """The drone's moment coefficient."""
        return self._KM

    @property
    def n_propellers(self):
        """The number of propellers on the drone."""
        return self._n_propellers

    @property
    def COM_link_idx(self):
        """The index of the center-of-mass (COM) link of the drone."""
        return self._COM_link_idx

    @property
    def propellers_idx(self):
        """The indices of the drone's propeller links."""
        return self._propellers_link_idxs

    @property
    def propellers_spin(self):
        """The spin direction for each propeller."""
        return self._propellers_spin
