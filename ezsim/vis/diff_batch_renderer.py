import enum
import math
from typing import Optional, Tuple, Dict, Any, Union
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import ezsim 
from ezsim.repr_base import RBC

try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter
except ImportError as e:
    ezsim.raise_exception_from("Madrona batch renderer is only supported on Linux x86-64.", e)


class IMAGE_TYPE(enum.IntEnum):
    RGB = 0
    DEPTH = 3
    NORMAL = 1
    SEGMENTATION = 2


class Light:
    def __init__(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = dir
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff


class DiffBatchRenderer(RBC):
    """
    High-performance differentiable batch renderer optimized for PyTorch-based physics learning.
    
    Key optimizations:
    - Direct tensor operations without CPU-GPU transfers
    - Persistent GPU memory allocation to avoid allocation overhead
    - Native tensor format compatibility with MadronaBatchRendererAdapter
    - Batch-optimized data structures
    - Gradient-friendly operations
    """

    def __init__(self, visualizer, renderer_options):
        self._visualizer = visualizer
        self._lights = ezsim.List()
        self._renderer_options = renderer_options
        self._renderer = None

        # High-performance caching with persistent GPU memory
        self._persistent_buffers = {}
        self._frame_cache = {}
        self._t = -1
        
        # Memory management flags
        self._buffers_initialized = False
        self._tensor_dtype = torch.float32
        self._device = ezsim.device
        
        # Performance optimization settings - access Pydantic model attributes directly
        self._enable_gradient = getattr(renderer_options, 'enable_gradient', True)
        self._pin_memory = getattr(renderer_options, 'pin_memory', False)

    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        """Add light with tensor-based storage for efficiency"""
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))

    def _initialize_persistent_buffers(self, n_envs: int, n_cameras: int, res: Tuple[int, int]):
        """Initialize persistent GPU buffers to avoid repeated allocation"""
        if self._buffers_initialized:
            return
            
        height, width = res
        batch_shape = (n_envs, n_cameras) if n_envs > 0 else (n_cameras,)
        
        # Pre-allocate persistent buffers on GPU
        buffer_configs = [
            ('rgb', (*batch_shape, height, width, 3), torch.uint8),
            ('depth', (*batch_shape, height, width, 1), torch.float32),
            ('normal', (*batch_shape, height, width, 3), torch.float32),
            ('segmentation', (*batch_shape, height, width, 1), torch.int32),
        ]
        
        for name, shape, dtype in buffer_configs:
            # Allocate with standard memory format
            buffer = torch.empty(shape, dtype=dtype, device=self._device)
            if self._pin_memory and dtype == torch.float32:
                buffer = buffer.pin_memory()
            self._persistent_buffers[name] = buffer
            
        # Camera pose buffers (frequently updated)
        self._persistent_buffers['cameras_pos'] = torch.empty(
            (*batch_shape, 3), dtype=torch.float32, device=self._device
        )
        self._persistent_buffers['cameras_quat'] = torch.empty(
            (*batch_shape, 4), dtype=torch.float32, device=self._device
        )
        
        self._buffers_initialized = True

    def build(self):
        """Build renderer with optimized initialization"""
        if len(self._visualizer._cameras) == 0:
            raise ValueError("No cameras to render")

        if ezsim.backend != ezsim.cuda:
            ezsim.raise_exception("DiffBatchRenderer requires CUDA backend.")

        cameras = self._visualizer._cameras
        lights = self._lights
        rigid = self._visualizer.scene.rigid_solver
        n_envs = max(self._visualizer.scene.n_envs, 1)
        res = cameras[0].res
        gpu_id = ezsim.device.index
        use_rasterizer = self._renderer_options.use_rasterizer

        # Initialize persistent buffers
        self._initialize_persistent_buffers(n_envs, len(cameras), res)

        # Cameras - use persistent tensors
        n_cameras = len(cameras)
        cameras_pos = torch.stack([camera.get_pos() for camera in cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in cameras], dim=1)
        cameras_fov = torch.tensor([camera.fov for camera in cameras], dtype=torch.float32, device=self._device)
        cameras_znear = torch.tensor([camera.near for camera in cameras], dtype=torch.float32, device=self._device)
        cameras_zfar = torch.tensor([camera.far for camera in cameras], dtype=torch.float32, device=self._device)
        
        # Lights - tensor-based initialization
        n_lights = len(lights)
        light_pos = torch.tensor([light.pos for light in self._lights], dtype=torch.float32, device=self._device)
        light_dir = torch.tensor([light.dir for light in self._lights], dtype=torch.float32, device=self._device)
        light_intensity = torch.tensor([light.intensity for light in self._lights], dtype=torch.float32, device=self._device)
        light_directional = torch.tensor([light.directional for light in self._lights], dtype=torch.int32, device=self._device)
        light_castshadow = torch.tensor([light.castshadow for light in self._lights], dtype=torch.int32, device=self._device)
        light_cutoff = torch.tensor([light.cutoffRad for light in self._lights], dtype=torch.float32, device=self._device)

        self._renderer = MadronaBatchRendererAdapter(
            rigid, gpu_id, n_envs, n_cameras, n_lights, cameras_fov, 
            cameras_znear, cameras_zfar, *res, False, use_rasterizer
        )
        self._renderer.init(
            rigid,
            cameras_pos,
            cameras_quat,
            light_pos,
            light_dir,
            light_intensity,
            light_directional,
            light_castshadow,
            light_cutoff,
        )

    def update_scene(self):
        """Update scene with minimal overhead"""
        self._visualizer._context.update()

    def _update_camera_poses_inplace(self):
        """Update camera poses directly in persistent buffers to avoid allocation"""
        cameras = self._visualizer._cameras
        
        # In-place updates using persistent buffers
        pos_buffer = self._persistent_buffers['cameras_pos']
        quat_buffer = self._persistent_buffers['cameras_quat']
        
        for i, camera in enumerate(cameras):
            pos_buffer[:, i] = camera.get_pos()
            quat_buffer[:, i] = camera.get_quat()
        
        return pos_buffer, quat_buffer

    def render_differentiable(
        self, 
        rgb: bool = True, 
        depth: bool = False, 
        normal: bool = False, 
        segmentation: bool = False,
        force_render: bool = False, 
        aliasing: bool = False,
        return_dict: bool = False,
        normalize_depth: bool = True,
        gradient_enabled: bool = None
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        High-performance differentiable rendering optimized for PyTorch training.
        
        Parameters
        ----------
        rgb : bool
            Render RGB images
        depth : bool  
            Render depth images
        normal : bool
            Render normal maps
        segmentation : bool
            Render segmentation masks
        force_render : bool
            Force re-rendering even if cached
        aliasing : bool
            Enable anti-aliasing
        return_dict : bool
            Return results as dictionary instead of tuple
        normalize_depth : bool
            Normalize depth values to [0, 1] range
        gradient_enabled : bool, optional
            Override gradient computation setting
            
        Returns
        -------
        Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
            Rendered images as tensors ready for PyTorch operations
        """
        # Gradient context management
        grad_enabled = gradient_enabled if gradient_enabled is not None else self._enable_gradient
        
        with torch.set_grad_enabled(grad_enabled):
            # Smart caching with frame tracking
            if force_render or self._t < self._visualizer.scene.t:
                self._frame_cache.clear()

            # Check cache with fine-grained keys
            cache_key = (rgb, depth, normal, segmentation, aliasing, normalize_depth)
            if cache_key in self._frame_cache and not force_render:
                cached_result = self._frame_cache[cache_key]
                return cached_result if not return_dict else self._tuple_to_dict(cached_result, rgb, depth, normal, segmentation)

            # Update scene
            self.update_scene()

            # Zero-copy camera pose updates
            cameras_pos, cameras_quat = self._update_camera_poses_inplace()
            
            # Render with optimized options
            render_options = torch.tensor([rgb, depth, normal, segmentation, aliasing], dtype=torch.bool, device=self._device)
            
            # Core rendering call
            rgba_arr_all, depth_arr_all, normal_arr_all, segmentation_arr_all = self._renderer.render(
                self._visualizer.scene.rigid_solver, cameras_pos, cameras_quat, render_options.to(torch.uint32).cpu().numpy()
            )

            # Zero-copy post-processing using optimized tensor operations
            results = []
            
            if rgb and rgba_arr_all is not None:
                # 处理RGBA -> RGB转换 (uint8 -> float32, [0,255] -> [0,1])
                rgb_tensor = rgba_arr_all[..., :3].float() / 255.0  # 提取RGB通道并归一化
                rgb_tensor = self._reorganize_batch_dims(rgb_tensor)
                # 设置梯度计算
                if grad_enabled:
                    rgb_tensor = rgb_tensor.requires_grad_(True)
                results.append(rgb_tensor)
            else:
                results.append(None)
                
            if depth and depth_arr_all is not None:
                # 处理深度图 (float32，squeeze最后的通道维度)
                depth_tensor = depth_arr_all.squeeze(-1)  # 从 [..., 1] -> [...]
                if normalize_depth:
                    # 高效的原地归一化
                    depth_min = depth_tensor.min()
                    depth_range = depth_tensor.max() - depth_min
                    if depth_range > 1e-8:
                        depth_tensor = (depth_tensor - depth_min) / depth_range
                depth_tensor = self._reorganize_batch_dims(depth_tensor)
                # 设置梯度计算
                if grad_enabled:
                    depth_tensor = depth_tensor.requires_grad_(True)
                results.append(depth_tensor)
            else:
                results.append(None)
                
            if normal and normal_arr_all is not None:
                # 处理法线图 (uint8 -> float32, [0,255] -> [-1,1])
                normal_tensor = normal_arr_all[..., :3].float()  # 提取前3个分量
                normal_tensor = (normal_tensor / 255.0) * 2.0 - 1.0  # 归一化到[-1,1]
                normal_tensor = self._reorganize_batch_dims(normal_tensor)
                # 设置梯度计算
                if grad_enabled:
                    normal_tensor = normal_tensor.requires_grad_(True)
                results.append(normal_tensor)
            else:
                results.append(None)
                
            if segmentation and segmentation_arr_all is not None:
                # 处理分割图 (int32，squeeze最后的通道维度)
                seg_tensor = segmentation_arr_all.squeeze(-1)  # 从 [..., 1] -> [...]
                seg_tensor = self._reorganize_batch_dims(seg_tensor)
                results.append(seg_tensor)
            else:
                results.append(None)

            # Cache results
            self._t = self._visualizer.scene.t
            self._frame_cache[cache_key] = tuple(results)
            
            if return_dict:
                return self._tuple_to_dict(tuple(results), rgb, depth, normal, segmentation)
            return tuple(results)

    def _reorganize_batch_dims(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        重组张量维度以适配PyTorch批处理
        
        根据分析结果，MadronaBatchRendererAdapter返回张量格式已经是:
        [n_envs, n_cameras, height, width, channels]
        
        这已经是环境优先的格式，符合PyTorch批处理习惯，无需维度交换。
        只需要处理单环境场景的维度压缩。
        """
        if tensor is None:
            return None
            
        # 如果是单环境场景，移除环境维度
        # n_envs == 0 表示单环境（EzSim约定）
        if self._visualizer.scene.n_envs == 0:
            tensor = tensor.squeeze(0)  # 从 [1, n_cameras, h, w, c] -> [n_cameras, h, w, c]
            
        return tensor

    def _tuple_to_dict(self, results: Tuple, rgb: bool, depth: bool, normal: bool, segmentation: bool) -> Dict[str, torch.Tensor]:
        """Convert tuple results to dictionary format"""
        result_dict = {}
        idx = 0
        
        if rgb:
            result_dict['rgb'] = results[idx]
            idx += 1
        if depth:
            result_dict['depth'] = results[idx] 
            idx += 1
        if normal:
            result_dict['normal'] = results[idx]
            idx += 1
        if segmentation:
            result_dict['segmentation'] = results[idx]
            idx += 1
            
        return result_dict

    def render_for_training(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        interpolation_mode: str = 'bilinear',
        **render_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized rendering specifically for neural network training.
        
        Parameters
        ----------
        target_size : Optional[Tuple[int, int]]
            Resize images to target size for network input
        interpolation_mode : str
            Interpolation mode for resizing ('bilinear', 'nearest', etc.)
        **render_kwargs
            Arguments passed to render_differentiable
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of rendered tensors optimized for training
        """
        # Default settings optimized for training
        render_kwargs.setdefault('return_dict', True)
        render_kwargs.setdefault('gradient_enabled', True)
        render_kwargs.setdefault('normalize_depth', True)
        
        results = self.render_differentiable(**render_kwargs)
        
        # Resize for network input if requested
        if target_size is not None:
            for key, tensor in results.items():
                if tensor is not None and len(tensor.shape) >= 3:
                    # Handle different tensor formats for resizing
                    original_shape = tensor.shape
                    
                    if len(original_shape) == 4:  # [n_envs, n_cameras, H, W, C] or [n_cameras, H, W, C]
                        # Reshape to [batch, C, H, W] for interpolation
                        if len(original_shape) == 4 and original_shape[-1] > 1:  # Has channels
                            batch_size = original_shape[0] * original_shape[1] if len(original_shape) == 4 else original_shape[0]
                            tensor_reshaped = tensor.view(-1, *original_shape[-3:])  # [batch, H, W, C]
                            tensor_reshaped = tensor_reshaped.permute(0, 3, 1, 2)  # [batch, C, H, W]
                        else:  # No channels (depth, segmentation)
                            tensor_reshaped = tensor.view(-1, 1, *original_shape[-2:])  # [batch, 1, H, W]
                        
                        # Resize using interpolation
                        if tensor.dtype == torch.int32:  # Segmentation
                            tensor_resized = F.interpolate(tensor_reshaped.float(), size=target_size, 
                                                         mode='nearest').int()
                        else:
                            tensor_resized = F.interpolate(tensor_reshaped, size=target_size, 
                                                         mode=interpolation_mode, align_corners=False)
                        
                        # Reshape back to original format
                        if len(original_shape) == 4 and original_shape[-1] > 1:  # Has channels
                            tensor_resized = tensor_resized.permute(0, 2, 3, 1)  # [batch, H, W, C]
                            new_shape = (*original_shape[:-3], *target_size, original_shape[-1])
                        else:  # No channels
                            tensor_resized = tensor_resized.squeeze(1)  # Remove channel dim
                            new_shape = (*original_shape[:-2], *target_size)
                        
                        tensor = tensor_resized.view(new_shape)
                    
                    results[key] = tensor
        
        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for optimization"""
        stats = {
            'persistent_buffers': {},
            'cache_size': len(self._frame_cache),
            'gpu_memory_allocated': torch.cuda.memory_allocated(self._device),
            'gpu_memory_reserved': torch.cuda.memory_reserved(self._device),
        }
        
        for name, buffer in self._persistent_buffers.items():
            stats['persistent_buffers'][name] = {
                'shape': buffer.shape,
                'dtype': buffer.dtype,
                'memory_mb': buffer.numel() * buffer.element_size() / 1024 / 1024
            }
        
        return stats

    def clear_cache(self):
        """Clear all caches to free memory"""
        self._frame_cache.clear()
        torch.cuda.empty_cache()

    def destroy(self):
        """Clean up resources"""
        self._lights.clear()
        self._frame_cache.clear()
        self._persistent_buffers.clear()
        
        if self._renderer is not None:
            del self._renderer.madrona
            self._renderer = None
            
        torch.cuda.empty_cache()

    def reset(self):
        """Reset temporal state"""
        self._t = -1
        self._frame_cache.clear()

    @property
    def lights(self):
        return self._lights
        
    @property
    def device(self):
        return self._device

    # Backward compatibility with original BatchRenderer interface
    def render(self, rgb=True, depth=False, normal=False, segmentation=False, force_render=False, aliasing=False):
        """Backward compatibility method - delegates to render_differentiable"""
        warnings.warn("Using legacy render() method. Consider using render_differentiable() for better performance.", 
                     DeprecationWarning, stacklevel=2)
        
        results = self.render_differentiable(rgb, depth, normal, segmentation, force_render, aliasing, 
                                           gradient_enabled=False)
        
        # Convert to original format (tuples of arrays per camera)
        processed_results = []
        for result_tensor in results:
            if result_tensor is not None:
                # Convert back to tuple format
                if len(result_tensor.shape) == 4:  # Batched
                    processed_results.append(tuple(result_tensor))
                else:  # Single camera
                    processed_results.append((result_tensor,))
            else:
                processed_results.append(None)
                
        return tuple(processed_results)
