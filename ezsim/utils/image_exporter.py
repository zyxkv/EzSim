import os
import cv2
import numpy as np
import torch
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import ezsim
from ezsim.utils.misc import tensor_to_array


def _export_frame_rgb_camera(i_env, export_dir, i_cam, i_step, rgb):
    # Take the rgb channel in case the rgb tensor has RGBA channel.
    rgb = np.flip(tensor_to_array(rgb[i_env, ..., :3]), axis=-1)
    cv2.imwrite(f"{export_dir}/rgb_cam{i_cam}_env{i_env}_{i_step:03d}.png", rgb)


def _export_frame_depth_camera(i_env, export_dir, i_cam, i_step, depth):
    depth = tensor_to_array(depth[i_env])
    cv2.imwrite(f"{export_dir}/depth_cam{i_cam}_env{i_env}_{i_step:03d}.png", depth)

def _export_frame_normal_camera(i_env, export_dir, i_cam, i_step, normal):
    normal = np.flip(tensor_to_array(normal[i_env,...,:3]), axis=-1)
    cv2.imwrite(f"{export_dir}/normal_cam{i_cam}_env{i_env}_{i_step:03d}.png", normal)

def index_to_rgb_color(index):
    """
    Map an integer index (0-255) to an RGB color.
    If index > 255, return white (255, 255, 255).
    """
    if index > 255:
        return (255, 255, 255)
    # Simple color palette: use a color map or generate colors
    # Here we use OpenCV's colormap for visualization
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    return tuple(int(x) for x in color_map[index, 0])

def segmentation_to_color(segmentation):
    """
    Convert a segmentation mask (H, W) with indices to an RGB image (H, W, 3).
    """
    # Ensure it's a 2D array
    if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
        segmentation = segmentation.squeeze(-1)
    elif segmentation.ndim != 2:
        raise ValueError(f"Expected 2D segmentation array, got shape {segmentation.shape}")
    
    h, w = segmentation.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Handle different data types
    if segmentation.dtype != np.int32:
        segmentation = segmentation.astype(np.int32)
    
    unique_indices = np.unique(segmentation)
    for idx in unique_indices:
        if idx >= 0:  # Skip invalid indices (like -1)
            color = index_to_rgb_color(int(idx))
            color_img[segmentation == idx] = color
    return color_img

def _export_frame_segmentation_camera(i_env, export_dir, i_cam, i_step, segmentation):
    seg_array = tensor_to_array(segmentation[i_env])
    # Remove the last dimension if it's 1 (squeeze the channel dimension)
    if seg_array.ndim == 3 and seg_array.shape[-1] == 1:
        seg_array = seg_array.squeeze(-1)
    try:
        color_img = segmentation_to_color(seg_array)
        cv2.imwrite(f"{export_dir}/segmentation_cam{i_cam}_env{i_env}_{i_step:03d}.png", color_img)
    except Exception as e:
        ezsim.logger.error(f"Failed to convert segmentation to color image: {e}")
        ezsim.logger.error(f"Segmentation array shape: {seg_array.shape}")
        ezsim.logger.error(f"Segmentation array dtype: {seg_array.dtype}")
        ezsim.logger.error(f"Segmentation unique values: {np.unique(seg_array)}")

class FrameImageExporter:
    """
    This class enables exporting images from all cameras and all environments in batch and in parallel, unlike
    `Camera.(start|stop)_recording` API, which only allows for exporting images from a single camera and environment.
    """

    def __init__(self, export_dir, depth_clip_max=100, depth_scale="log"):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
        self.depth_scale = depth_scale

    def _normalize_depth(self, depth):
        """Normalize depth values for visualization.

        Args:
            depth: Tensor of depth values

        Returns:
            Normalized depth tensor as uint8
        """
        # Clip depth values
        depth = depth.clamp(0.0, self.depth_clip_max)

        # Apply scaling if specified
        if self.depth_scale == "log":
            depth = torch.log(depth + 1)

        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-3, -2), keepdim=True)
        depth_max = depth.amax(dim=(-3, -2), keepdim=True)

        # Normalize to 0-255 range
        return torch.where(
            depth_max - depth_min > ezsim.EPS, ((depth_max - depth) / (depth_max - depth_min) * 255).to(torch.uint8), 0
        )

    def export_frame_all_cameras(self, i_step, camera_idx=None, rgb=None, depth=None, normal=None, segmentation=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: rgb image is a sequence of tensors of shape (n_envs, H, W, 3).
            depth: Depth image is a sequence of tensors of shape (n_envs, H, W).
            normal: Normal image is a sequence of tensors of shape (n_envs, H, W, 3).
            segmentation: Segmentation image is a sequence of tensors of shape (n_envs, H, W).
        """
        if rgb is None and depth is None:
            ezsim.logger.info("No rgb or depth images to export")
            return
        if rgb is not None and (not isinstance(rgb, (tuple, list)) or not rgb):
            ezsim.raise_exception("'rgb' must be a non-empty sequence of tensors.")
        if depth is not None and (not isinstance(depth, (tuple, list)) or not depth):
            ezsim.raise_exception("'depth' must be a non-empty sequence of tensors.")
        if normal is not None and (not isinstance(normal, (tuple, list)) or not normal):   
            ezsim.raise_exception("'normal' must be a non-empty sequence of tensors.")
        if segmentation is not None and (not isinstance(segmentation, (tuple, list)) or not segmentation):
            ezsim.raise_exception("'segmentation' must be a non-empty sequence of tensors.")    
        if camera_idx is None:
            camera_idx = range(len(depth or rgb))
        for i_cam in camera_idx:
            rgb_cam, depth_cam, normal_cam, segmentation_cam = None, None, None, None
            if rgb is not None:
                rgb_cam = rgb[i_cam]
            if depth is not None:
                depth_cam = depth[i_cam]
            if normal is not None:
                normal_cam = normal[i_cam]
            if segmentation is not None:
                segmentation_cam = segmentation[i_cam]
            self.export_frame_single_camera(i_step, i_cam, rgb_cam, depth_cam, normal_cam, segmentation_cam)

    def export_frame_single_camera(self, i_step, i_cam, rgb=None, depth=None, normal=None, segmentation=None):

        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            i_cam: The index of the camera.
            rgb: rgb image tensor of shape (n_envs, H, W, 3).
            depth: Depth tensor of shape (n_envs, H, W).
        """
        if rgb is not None:
            rgb = torch.as_tensor(rgb, dtype=torch.uint8, device=ezsim.device)

            # Unsqueeze rgb to (n_envs, H, W, 3)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            if rgb.ndim != 4 or rgb.shape[-1] != 3:
                ezsim.raise_exception("'rgb' must be a tensor of shape (n_envs, H, W, 3)")

            rgb_job = partial(
                _export_frame_rgb_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                rgb=rgb,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(rgb_job, np.arange(len(rgb)))

        if depth is not None:
            depth = torch.as_tensor(depth, dtype=torch.float32, device=ezsim.device)

            # Unsqueeze depth to (n_envs, H, W, 1)
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 2:
                depth = depth.reshape((1, *depth.shape, 1))
            depth = self._normalize_depth(depth)
            if depth.ndim != 4 or depth.shape[-1] != 1:
                ezsim.raise_exception("'rgb' must be a tensor of shape (n_envs, H, W, 1)")

            depth_job = partial(
                _export_frame_depth_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                depth=depth,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(depth_job, np.arange(len(depth)))
        
        if normal is not None:
            normal = torch.as_tensor(normal, dtype=torch.float32, device=ezsim.device)

            # Unsqueeze normal to (n_envs, H, W, 3)
            if normal.ndim == 3:
                normal = normal.unsqueeze(0)
            if normal.ndim != 4 or normal.shape[-1] != 3:
                ezsim.raise_exception("'normal' must be a tensor of shape (n_envs, H, W, 3)")

            normal_job = partial(
                _export_frame_normal_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                normal=normal,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(normal_job, np.arange(len(normal)))

        if segmentation is not None:    
            segmentation = torch.as_tensor(segmentation, dtype=torch.int32, device=ezsim.device)

            # Unsqueeze segmentation to (n_envs, H, W,1)
            if segmentation.ndim == 3:
                segmentation = segmentation.unsqueeze(0)
            elif segmentation.ndim == 2:
                segmentation = segmentation.reshape((1, *segmentation.shape, 1))

            if segmentation.ndim != 4 or segmentation.shape[-1] != 1:
                ezsim.raise_exception("'segmentation' must be a tensor of shape (n_envs, H, W, 1)")

            segmentation_job = partial(
                _export_frame_segmentation_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                segmentation=segmentation,
            )   

            with ThreadPoolExecutor() as executor:
                executor.map(segmentation_job, np.arange(len(segmentation)))


class VideoExporter:
    """
    This class enables exporting video from a single camera.
    """

    def __init__(self, filename, fps=30, codec="mp4v"):
        self.filename = filename
        self.fps = fps
        self.codec = codec
        self.writer = None

    def start(self, res):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, res)

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def export_frame_all_cameras(self):
        pass 

    def export_frame_single_camera(self):
        pass 