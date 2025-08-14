from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import taichi as ti
import torch

import ezsim
from ezsim.engine.entities import RigidEntity
from ezsim.utils.geom import inv_transform_by_quat, ti_inv_transform_by_quat, transform_by_trans_quat
from ezsim.utils.misc import tensor_to_array

from .base_sensor import Sensor


@dataclass
class IMUData:
    """IMU sensor data structure"""
    lin_acc_body: torch.Tensor    # Linear acceleration in body frame (m/s²)
    ang_vel_body: torch.Tensor    # Angular velocity in body frame (rad/s)
    ang_acc_body: torch.Tensor    # Angular acceleration in body frame (rad/s²)
    lin_vel_body: torch.Tensor    # Linear velocity in body frame (m/s)
    timestamp: float              # Simulation timestamp


@dataclass
class IMUNoiseConfig:
    """IMU noise configuration parameters"""
    # Accelerometer noise parameters
    accel_noise_density: float = 0.01      # m/s²/√Hz
    accel_bias_stability: float = 0.001    # m/s²
    accel_random_walk: float = 0.0001      # m/s²/√s
    
    # Gyroscope noise parameters
    gyro_noise_density: float = 0.001      # rad/s/√Hz
    gyro_bias_stability: float = 0.0001    # rad/s
    gyro_random_walk: float = 0.00001      # rad/s/√s
    
    # General parameters
    enable_noise: bool = True
    seed: int = 42


@ti.data_oriented
class IMUSensor(Sensor):
    """
    Inertial Measurement Unit (IMU) sensor for rigid body simulation.
    
    Provides linear acceleration, angular velocity, angular acceleration, and linear velocity
    measurements in the body frame, with optional realistic noise modeling.

    Parameters
    ----------
    entity : RigidEntity
        The entity to which this sensor is attached.
    link_idx : int, optional
        The index of the link to which this sensor is attached. If None, defaults to the base link.
    noise_config : IMUNoiseConfig, optional
        Noise configuration for realistic sensor simulation.
    update_rate : float, optional
        Sensor update rate in Hz. Default is 1000.0 Hz.
    """
    
    _last_imu_update_step = -1
    _cached_data: Optional[IMUData] = None

    def __init__(self, 
                 entity: RigidEntity, 
                 link_idx: Optional[int] = None,
                 noise_config: Optional[IMUNoiseConfig] = None,
                 update_rate: float = 1000.0):
        
        self._cls = self.__class__
        self._entity = entity
        self._sim = entity._sim
        self.link_idx = link_idx if link_idx is not None else entity.base_link_idx
        
        # Noise configuration
        self.noise_config = noise_config if noise_config is not None else IMUNoiseConfig(enable_noise=False)
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        # Initialize noise states if noise is enabled
        if self.noise_config.enable_noise:
            self._init_noise_states()
    
    def _init_noise_states(self):
        """Initialize noise-related internal states"""
        # Set random seed for reproducibility
        torch.manual_seed(self.noise_config.seed)
        np.random.seed(self.noise_config.seed)
        
        # Initialize bias states (these will evolve over time)
        self._accel_bias = torch.zeros(3, device=ezsim.device, dtype=ezsim.tc_float)
        self._gyro_bias = torch.zeros(3, device=ezsim.device, dtype=ezsim.tc_float)
        
        # Previous values for bias random walk
        self._prev_accel_bias = torch.zeros_like(self._accel_bias)
        self._prev_gyro_bias = torch.zeros_like(self._gyro_bias)

    def read(self, envs_idx: Optional[List[int]] = None) -> IMUData:
        """
        Read IMU sensor data.
        
        Parameters
        ----------
        envs_idx : Optional[List[int]]
            Environment indices to read data for. If None, reads for all environments.
            
        Returns
        -------
        IMUData
            IMU sensor measurements in body frame.
        """
        # Check if we need to update the cache
        if self._cls._last_imu_update_step == self._sim.cur_step_global and self._cls._cached_data is not None:
            cached_data = self._cls._cached_data
            if envs_idx is not None:
                # Return subset of cached data
                return IMUData(
                    lin_acc_body=cached_data.lin_acc_body[envs_idx],
                    ang_vel_body=cached_data.ang_vel_body[envs_idx],
                    ang_acc_body=cached_data.ang_acc_body[envs_idx],
                    lin_vel_body=cached_data.lin_vel_body[envs_idx],
                    timestamp=cached_data.timestamp
                )
            return cached_data

        # Update cache
        self._cls._last_imu_update_step = self._sim.cur_step_global
        
        # Get raw sensor data
        links_idx = [self.link_idx]
        
        # Linear acceleration in body frame (with gravity compensation) 
        lin_acc_body = self._sim.rigid_solver.get_links_acc(links_idx=links_idx, mimick_imu=True)
        
        # Use new optimized functions for body frame data if available
        try:
            # Angular velocity in body frame (direct)
            ang_vel_body = self._sim.rigid_solver.get_links_ang_body(links_idx=links_idx)
            
            # Angular acceleration in body frame (direct)
            ang_acc_body = self._sim.rigid_solver.get_links_acc_ang_body(links_idx=links_idx)
            
            # Linear velocity in body frame (direct)
            lin_vel_body = self._sim.rigid_solver.get_links_vel_body(links_idx=links_idx, ref="link_origin")
            
        except AttributeError:
            # Fallback to world frame + manual transformation if new functions not available
            ang_vel_world = self._sim.rigid_solver.get_links_ang(links_idx=links_idx)
            link_quat = self._sim.rigid_solver.get_links_quat(links_idx=links_idx)
            ang_vel_body = inv_transform_by_quat(ang_vel_world, link_quat)
            
            # Angular acceleration in world frame, then convert to body frame
            ang_acc_world = self._sim.rigid_solver.get_links_acc_ang(links_idx=links_idx)
            ang_acc_body = inv_transform_by_quat(ang_acc_world, link_quat)
            
            # Linear velocity in world frame, then convert to body frame
            lin_vel_world = self._sim.rigid_solver.get_links_vel(links_idx=links_idx, ref="link_origin")
            lin_vel_body = inv_transform_by_quat(lin_vel_world, link_quat)
        
        # Apply noise if enabled
        if self.noise_config.enable_noise:
            lin_acc_body = self._add_accelerometer_noise(lin_acc_body)
            ang_vel_body = self._add_gyroscope_noise(ang_vel_body)
            # Note: We don't typically add noise to angular acceleration and linear velocity
            # as they are derived quantities, but could be added if needed
        
        # Create IMU data structure
        imu_data = IMUData(
            lin_acc_body=lin_acc_body.squeeze(0) if lin_acc_body.shape[0] == 1 else lin_acc_body,
            ang_vel_body=ang_vel_body.squeeze(0) if ang_vel_body.shape[0] == 1 else ang_vel_body,
            ang_acc_body=ang_acc_body.squeeze(0) if ang_acc_body.shape[0] == 1 else ang_acc_body,
            lin_vel_body=lin_vel_body.squeeze(0) if lin_vel_body.shape[0] == 1 else lin_vel_body,
            timestamp=self._sim.cur_step_global * self._sim._substep_dt
        )
        
        # Cache the data
        self._cls._cached_data = imu_data
        
        # Return subset if requested
        if envs_idx is not None:
            return IMUData(
                lin_acc_body=imu_data.lin_acc_body[envs_idx],
                ang_vel_body=imu_data.ang_vel_body[envs_idx],
                ang_acc_body=imu_data.ang_acc_body[envs_idx],
                lin_vel_body=imu_data.lin_vel_body[envs_idx],
                timestamp=imu_data.timestamp
            )
        
        return imu_data
    
    def _add_accelerometer_noise(self, clean_data: torch.Tensor) -> torch.Tensor:
        """Add realistic accelerometer noise to clean data"""
        noisy_data = clean_data.clone()
        
        # White noise
        white_noise = torch.randn_like(clean_data) * self.noise_config.accel_noise_density / np.sqrt(self.dt)
        
        # Bias instability (random walk)
        bias_drift = torch.randn_like(clean_data) * self.noise_config.accel_random_walk * np.sqrt(self.dt)
        self._accel_bias += bias_drift
        
        # Apply noise
        noisy_data += white_noise + self._accel_bias
        
        return noisy_data
    
    def _add_gyroscope_noise(self, clean_data: torch.Tensor) -> torch.Tensor:
        """Add realistic gyroscope noise to clean data"""
        noisy_data = clean_data.clone()
        
        # White noise  
        white_noise = torch.randn_like(clean_data) * self.noise_config.gyro_noise_density / np.sqrt(self.dt)
        
        # Bias instability (random walk)
        bias_drift = torch.randn_like(clean_data) * self.noise_config.gyro_random_walk * np.sqrt(self.dt)
        self._gyro_bias += bias_drift
        
        # Apply noise
        noisy_data += white_noise + self._gyro_bias
        
        return noisy_data
    
    def get_linear_acceleration(self, envs_idx: Optional[List[int]] = None) -> torch.Tensor:
        """Get only linear acceleration measurement"""
        return self.read(envs_idx).lin_acc_body
    
    def get_angular_velocity(self, envs_idx: Optional[List[int]] = None) -> torch.Tensor:
        """Get only angular velocity measurement"""
        return self.read(envs_idx).ang_vel_body
    
    def get_angular_acceleration(self, envs_idx: Optional[List[int]] = None) -> torch.Tensor:
        """Get only angular acceleration measurement"""
        return self.read(envs_idx).ang_acc_body
    
    def get_linear_velocity(self, envs_idx: Optional[List[int]] = None) -> torch.Tensor:
        """Get only linear velocity measurement"""
        return self.read(envs_idx).lin_vel_body
    
    def reset_noise_states(self):
        """Reset noise-related states (useful for episode resets)"""
        if self.noise_config.enable_noise:
            self._accel_bias.zero_()
            self._gyro_bias.zero_()
            self._prev_accel_bias.zero_()
            self._prev_gyro_bias.zero_()
    
    def get_data(self, envs_idx: Optional[List[int]] = None) -> IMUData:
        """
        Alias for read() method for backward compatibility.
        
        Returns
        -------
        IMUData
            Complete IMU sensor data structure.
        """
        return self.read(envs_idx)



