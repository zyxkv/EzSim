from typing import Literal
import inspect
import os
import time

import cv2
import numpy as np

import ezsim
import ezsim.utils.geom as gu
from ezsim.sensors import Sensor
from ezsim.utils.misc import tensor_to_array


class Camera(Sensor):
    """
    A camera which can be used to render RGB, depth, and segmentation images.
    Supports either rasterizer or raytracer for rendering, specified by `scene.renderer`.

    Parameters
    ----------
    visualizer : ezsim.Visualizer
        The visualizer object that the camera is associated with.
    idx : int
        The index of the camera.
    model : str
        Specifies the camera model. Options are 'pinhole' or 'thinlens'.
    res : tuple of int, shape (2,)
        The resolution of the camera, specified as a tuple (width, height).
    pos : tuple of float, shape (3,)
        The position of the camera in the scene, specified as (x, y, z).
    lookat : tuple of float, shape (3,)
        The point in the scene that the camera is looking at, specified as (x, y, z).
    up : tuple of float, shape (3,)
        The up vector of the camera, defining its orientation, specified as (x, y, z).
    fov : float
        The vertical field of view of the camera in degrees.
    aperture : float
        The aperture size of the camera, controlling depth of field.
    focus_dist : float | None
        The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`.
    GUI : bool
        Whether to display the camera's rendered image in a separate GUI window.
    spp : int, optional
        Samples per pixel. Only available when using the RayTracer renderer. Defaults to 256.
    denoise : bool
        Whether to denoise the camera's rendered image. Only available when using the RayTracer renderer.
        Defaults to True.  If OptiX denoiser is not available on your platform, consider enabling the OIDN denoiser
        option when building RayTracer.
    near : float
        The near plane of the camera.
    far : float
        The far plane of the camera.
    transform : np.ndarray, shape (4, 4), optional
        The transform matrix of the camera.
    """

    def __init__(
        self,
        visualizer,
        idx=0,
        model="pinhole",  # pinhole or thinlens
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
        aperture=2.8,
        focus_dist=None,
        GUI=False,
        spp=256,
        denoise=True,
        near=0.05,
        far=100.0,
        transform=None,
    ):
        self._idx = idx
        self._uid = ezsim.UID()
        self._model = model
        self._res = res
        self._fov = fov
        self._aperture = aperture
        self._focus_dist = focus_dist
        self._GUI = GUI
        self._spp = spp
        self._denoise = denoise
        self._near = near
        self._far = far
        self._pos = pos
        self._lookat = lookat
        self._up = up
        self._transform = transform
        self._aspect_ratio = self._res[0] / self._res[1]
        self._visualizer = visualizer
        self._is_built = False
        self._attached_link = None
        self._attached_offset_T = None
        self._attached_env_idx = None

        self._in_recording = False
        self._recorded_imgs = []
        # DEBUG: recorded depth, segmentation, and normal images
        self._recorded_depths = []
        self._recorded_segmentations = []
        self._recorded_normals = []

        self._init_pos = np.array(pos)

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None

        if self._model not in ["pinhole", "thinlens"]:
            ezsim.raise_exception(f"Invalid camera model: {self._model}")

        if self._focus_dist is None:
            self._focus_dist = np.linalg.norm(np.array(lookat) - np.array(pos))

    def _build(self):
        self._rasterizer = self._visualizer.rasterizer
        self._raytracer = self._visualizer.raytracer

        self._rgb_stacked = self._visualizer._context.env_separate_rigid
        self._other_stacked = self._visualizer._context.env_separate_rigid

        if self._rasterizer is not None:
            self._rasterizer.add_camera(self)
        if self._raytracer is not None:
            self._raytracer.add_camera(self)
            self._rgb_stacked = False  # TODO: Raytracer currently does not support batch rendering

        self._is_built = True
        self.set_pose(self._transform, self._pos, self._lookat, self._up)

    def attach(self, rigid_link, offset_T, env_idx: int | None = None):
        """
        Attach the camera to a rigid link in the scene.

        Once attached, the camera's position and orientation can be updated relative to the attached link using `move_to_attach()`. This is useful for mounting the camera to dynamic entities like robots or articulated objects.

        Parameters
        ----------
        rigid_link : ezsim.RigidLink
            The rigid link to which the camera should be attached.
        offset_T : np.ndarray, shape (4, 4)
            The transformation matrix specifying the camera's pose relative to the rigid link.
        env_idx : int
            The environment index this camera should be tied to. Offsets the `offset_T` accordingly. Must be specified
            if running parallel environments

        Raises
        ------
        Exception
            If running parallel simulations but env_idx is not specified.
        Exception
            If invalid env_idx is specified (env_idx >= n_envs)
        """
        self._attached_link = rigid_link
        self._attached_offset_T = offset_T
        if self._visualizer._scene.n_envs > 0 and env_idx is None:
            ezsim.raise_exception("Must specify env_idx when running parallel simulations")
        if env_idx is not None:
            n_envs = self._visualizer._scene.n_envs
            if env_idx >= n_envs:
                ezsim.raise_exception(f"Invalid env_idx {env_idx} for camera, configured for {n_envs} environments")
            self._attached_env_idx = env_idx

    def detach(self):
        """
        Detach the camera from the currently attached rigid link.

        After detachment, the camera will stop following the motion of the rigid link and maintain its current world pose. Calling this method has no effect if the camera is not currently attached.
        """
        self._attached_link = None
        self._attached_offset_T = None
        self._attached_env_idx = None

    @ezsim.assert_built
    def move_to_attach(self):
        """
        Move the camera to follow the currently attached rigid link.

        This method updates the camera's pose using the transform of the attached rigid link combined with the specified offset. It should only be called after `attach()` has been used. This method is not compatible with simulations running multiple environments in parallel.

        Raises
        ------
        Exception
            If the camera has not been mounted using `attach()`.
        """
        if self._attached_link is None:
            ezsim.raise_exception(f"The camera hasn't been mounted!")

        link_pos = tensor_to_array(self._attached_link.get_pos(envs_idx=self._attached_env_idx))
        link_quat = tensor_to_array(self._attached_link.get_quat(envs_idx=self._attached_env_idx))
        if self._attached_env_idx is not None:
            link_pos = link_pos[0] + self._visualizer._scene.envs_offset[self._attached_env_idx]
            link_quat = link_quat[0]
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        transform = link_T @ self._attached_offset_T
        self.set_pose(transform=transform)

    @ezsim.assert_built
    def read(self):
        """
        Obtain the RGB camera view.
        This is a temporary implementation to make Camera a Sensor.
        """
        rgb, _, _, _ = self.render()
        return rgb

    @ezsim.assert_built
    def render(self, rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False):
        """
        Render the camera view. Note that the segmentation mask can be colorized, and if not colorized, it will store an object index in each pixel based on the segmentation level specified in `VisOptions.segmentation_level`. For example, if `segmentation_level='link'`, the segmentation mask will store `link_idx`, which can then be used to retrieve the actual link objects using `scene.rigid_solver.links[link_idx]`.
        If `env_separate_rigid` in `VisOptions` is set to True, each component will return a stack of images, with the number of images equal to `len(rendered_envs_idx)`.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render RGB image(s).
        depth : bool, optional
            Whether to render depth image(s).
        segmentation : bool, optional
            Whether to render the segmentation mask(s).
        colorize_seg : bool, optional
            If True, the segmentation mask will be colorized.
        normal : bool, optional
            Whether to render the surface normal.

        Returns
        -------
        rgb_arr : np.ndarray
            The rendered RGB image(s).
        depth_arr : np.ndarray
            The rendered depth image(s).
        seg_arr : np.ndarray
            The rendered segmentation mask(s).
        normal_arr : np.ndarray
            The rendered surface normal(s).
        """

        if (rgb or depth or segmentation or normal) is False:
            ezsim.raise_exception("Nothing to render.")

        rgb_arr, depth_arr, seg_idxc_arr, seg_arr, normal_arr = None, None, None, None, None

        if self._followed_entity is not None:
            self.update_following()

        if self._raytracer is not None:
            if rgb:
                self._raytracer.update_scene()
                rgb_arr = self._raytracer.render_camera(self)

            if depth or segmentation or normal:
                if self._rasterizer is not None:
                    self._rasterizer.update_scene()
                    _, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                        self, False, depth, segmentation, normal=normal
                    )
                else:
                    ezsim.raise_exception("Cannot render depth or segmentation image.")

        elif self._rasterizer is not None:
            self._rasterizer.update_scene()
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                self, rgb, depth, segmentation, normal=normal
            )

        else:
            ezsim.raise_exception("No renderer was found.")

        if seg_idxc_arr is not None:
            if colorize_seg or (self._GUI and self._visualizer.connected_to_display):
                seg_color_arr = self._rasterizer._context.colorize_seg_idxc_arr(seg_idxc_arr)
            if colorize_seg:
                seg_arr = seg_color_arr
            else:
                seg_arr = seg_idxc_arr

        depth_img = None
        # succeed rendering, and display image
        if self._GUI and self._visualizer.connected_to_display:
            title = f"EzSim - Camera {self._idx}"

            if rgb:
                rgb_img = rgb_arr[..., [2, 1, 0]]
                rgb_env = ""
                if self._rgb_stacked:
                    rgb_img = rgb_img[0]
                    rgb_env = " Environment 0"
                cv2.imshow(f"{title + rgb_env} [RGB]", rgb_img)

            other_env = " Environment 0" if self._other_stacked else ""
            if depth:
                depth_min = depth_arr.min()
                depth_max = depth_arr.max()
                depth_normalized = (depth_arr - depth_min) / (depth_max - depth_min)
                depth_normalized = 1 - depth_normalized  # closer objects appear brighter
                depth_img = (depth_normalized * 255).astype(np.uint8)
                if self._other_stacked:
                    depth_img = depth_img[0]
                if self._in_recording:
                   self._recorded_depths.append(depth_img[..., None].repeat(3, axis=-1))

                cv2.imshow(f"{title + other_env} [Depth]", depth_img)

            if segmentation:
                seg_img = seg_color_arr[..., [2, 1, 0]]
                if self._other_stacked:
                    seg_img = seg_img[0]

                cv2.imshow(f"{title + other_env} [Segmentation]", seg_img)

            if normal:
                normal_img = normal_arr[..., [2, 1, 0]]
                if self._other_stacked:
                    normal_img = normal_img[0]

                cv2.imshow(f"{title + other_env} [Normal]", normal_img)

            cv2.waitKey(1)

        if self._in_recording:
            if rgb_arr is not None:
                self._recorded_imgs.append(rgb_arr)
            if depth_arr is not None:
                if depth_img is None:
                    depth_min = depth_arr.min()
                    depth_max = depth_arr.max()
                    depth_normalized = (depth_arr - depth_min) / (depth_max - depth_min)
                    depth_img = (depth_normalized * 255).astype(np.uint8)
                    self._recorded_depths.append(depth_img[...,None].repeat(3, axis=-1))
            if seg_arr is not None and colorize_seg:
                self._recorded_segmentations.append(seg_color_arr[..., [2, 1, 0]])
            if normal_arr is not None:
                self._recorded_normals.append(normal_arr[..., [2, 1, 0]])


        return rgb_arr, depth_arr, seg_arr, normal_arr

    @ezsim.assert_built
    def get_segmentation_idx_dict(self):
        """
        Returns a dictionary mapping segmentation indices to scene entities.

        In the segmentation map:
        - Index 0 corresponds to the background (-1).
        - Indices > 0 correspond to scene elements, which may be represented as:
            - `entity_id`
            - `(entity_id, link_id)`
            - `(entity_id, link_id, geom_id)`
          depending on the material type and the configured segmentation level.
        """
        return self._rasterizer._context.seg_idxc_map

    @ezsim.assert_built
    def render_pointcloud(self, world_frame=True):
        """
        Render a partial point cloud from the camera view. Returns a (res[0], res[1], 3) numpy array representing the point cloud in each pixel.
        Parameters
        ----------
        world_frame : bool, optional
            Whether the point cloud is on camera frame or world frame.
        Returns
        -------
        pc : np.ndarray
            the point cloud
        mask_arr : np.ndarray
            The valid depth mask.
        """
        if self._rasterizer is not None:
            self._rasterizer.update_scene()
            rgb_arr, depth_arr, seg_idxc_arr, normal_arr = self._rasterizer.render_camera(
                self, False, True, False, normal=False
            )

            def opengl_projection_matrix_to_intrinsics(P: np.ndarray, width: int, height: int):
                """Convert OpenGL projection matrix to camera intrinsics.
                Args:
                    P (np.ndarray): OpenGL projection matrix.
                    width (int): Image width.
                    height (int): Image height
                Returns:
                    np.ndarray: Camera intrinsics. [3, 3]
                """

                fx = P[0, 0] * width / 2
                fy = P[1, 1] * height / 2
                cx = (1.0 - P[0, 2]) * width / 2
                cy = (1.0 + P[1, 2]) * height / 2

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                return K

            def backproject_depth_to_pointcloud(K: np.ndarray, depth: np.ndarray, pose, world, znear, zfar):
                """Convert depth image to pointcloud given camera intrinsics.
                Args:
                    depth (np.ndarray): Depth image.
                Returns:
                    np.ndarray: (x, y, z) Point cloud. [n, m, 3]
                """
                _fx = K[0, 0]
                _fy = K[1, 1]
                _cx = K[0, 2]
                _cy = K[1, 2]

                # Mask out invalid depth
                mask = np.where((depth > znear) & (depth < zfar * 0.99))
                # zfar * 0.99 for filtering out precision error of float
                height, width = depth.shape
                y, x = np.meshgrid(np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij")
                x = x.reshape((-1,))
                y = y.reshape((-1,))

                # Normalize pixel coordinates
                normalized_x = x - _cx
                normalized_y = y - _cy

                # Convert to world coordinates
                depth_grid = depth[y, x]
                world_x = normalized_x * depth_grid / _fx
                world_y = normalized_y * depth_grid / _fy
                world_z = depth_grid

                pc = np.stack((world_x, world_y, world_z), axis=1)

                point_cloud_h = np.concatenate((pc, np.ones((len(pc), 1), dtype=np.float32)), axis=1)
                if world:
                    point_cloud_world = point_cloud_h @ pose.T
                    point_cloud_world = point_cloud_world[:, :3].reshape((depth.shape[0], depth.shape[1], 3))
                    return point_cloud_world, mask
                else:
                    point_cloud = point_cloud_h[:, :3].reshape((depth.shape[0], depth.shape[1], 3))
                    return point_cloud, mask

            intrinsic_K = opengl_projection_matrix_to_intrinsics(
                self._rasterizer._camera_nodes[self.uid].camera.get_projection_matrix(),
                width=self.res[0],
                height=self.res[1],
            )

            T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
            cam_pose = self._rasterizer._camera_nodes[self.uid].matrix @ T_OPENGL_TO_OPENCV

            pc, mask = backproject_depth_to_pointcloud(
                intrinsic_K, depth_arr, cam_pose, world_frame, self.near, self.far
            )

            return pc, mask

        else:
            ezsim.raise_exception("We need a rasterizer to render depth and then convert it to pount cloud.")

    @ezsim.assert_built
    def set_pose(self, transform=None, pos=None, lookat=None, up=None):
        """
        Set the pose of the camera.
        Note that `transform` has a higher priority than `pos`, `lookat`, and `up`. If `transform` is provided, the camera pose will be set based on the transform matrix. Otherwise, the camera pose will be set based on `pos`, `lookat`, and `up`.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4), optional
            The transform matrix of the camera.
        pos : array-like, shape (3,), optional
            The position of the camera.
        lookat : array-like, shape (3,), optional
            The lookat point of the camera.
        up : array-like, shape (3,), optional
            The up vector of the camera.

        """
        if transform is not None:
            assert transform.shape == (4, 4)
            self._transform = transform
            self._pos, self._lookat, self._up = gu.T_to_pos_lookat_up(transform)

        else:
            if pos is not None:
                self._pos = pos

            if lookat is not None:
                self._lookat = lookat

            if up is not None:
                self._up = up

            self._transform = gu.pos_lookat_up_to_T(self._pos, self._lookat, self._up)

        if self._rasterizer is not None:
            self._rasterizer.update_camera(self)
        if self._raytracer is not None:
            self._raytracer.update_camera(self)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the camera to follow a specified entity.

        Parameters
        ----------
        entity : ezsim.Entity
            The entity to follow.
        fixed_axis : (float, float, float), optional
            The fixed axis for the camera's movement. For each axis, if None, the camera will move freely. If a float, the viewer will be fixed on at that value.
            For example, [None, None, None] will allow the camera to move freely while following, [None, None, 0.5] will fix the viewer's z-axis at 0.5.
        smoothing : float, optional
            The smoothing factor for the camera's movement. If None, no smoothing will be applied.
        fix_orientation : bool, optional
            If True, the camera will maintain its orientation relative to the world. If False, the camera will look at the base link of the entity.
        """
        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation

    @ezsim.assert_built
    def update_following(self):
        """
        Update the camera position to follow the specified entity.
        """

        entity_pos = self._followed_entity.get_pos()[0].cpu().numpy()
        if entity_pos.ndim > 1:  # check for multiple envs
            entity_pos = entity_pos[0]
        camera_pos = np.array(self._pos)
        camera_pose = np.array(self._transform)
        lookat_pos = np.array(self._lookat)

        if self._follow_smoothing is not None:
            # Smooth camera movement with a low-pass filter
            camera_pos = self._follow_smoothing * camera_pos + (1 - self._follow_smoothing) * (
                entity_pos + self._init_pos
            )
            lookat_pos = self._follow_smoothing * lookat_pos + (1 - self._follow_smoothing) * entity_pos
        else:
            camera_pos = entity_pos + self._init_pos
            lookat_pos = entity_pos

        for i, fixed_axis in enumerate(self._follow_fixed_axis):
            # Fix the camera's position along the specified axis
            if fixed_axis is not None:
                camera_pos[i] = fixed_axis

        if self._follow_fix_orientation:
            # Keep the camera orientation fixed by overriding the lookat point
            camera_pose[:3, 3] = camera_pos
            self.set_pose(transform=camera_pose)
        else:
            self.set_pose(pos=camera_pos, lookat=lookat_pos)

    @ezsim.assert_built
    def set_params(self, fov=None, aperture=None, focus_dist=None, intrinsics=None):
        """
        Update the camera parameters.

        Parameters
        ----------
        fov: float, optional
            The vertical field of view of the camera.
        aperture : float, optional
            The aperture of the camera. Only supports 'thinlens' camera model.
        focus_dist : float, optional
            The focus distance of the camera. Only supports 'thinlens' camera model.
        intrinsics : np.ndarray, shape (3, 3), optional
            The intrinsics matrix of the camera. If provided, it should be consistent with the specified 'fov'.
        """
        if self.model != "thinlens" and (aperture is not None or focus_dist is not None):
            ezsim.logger.warning("Only `thinlens` camera model supports parameter update.")

        if aperture is not None:
            if self.model != "thinlens":
                ezsim.logger.warning("Only `thinlens` camera model supports `aperture`.")
            self._aperture = aperture
        if focus_dist is not None:
            if self.model != "thinlens":
                ezsim.logger.warning("Only `thinlens` camera model supports `focus_dist`.")
            self._focus_dist = focus_dist

        if fov is not None:
            self._fov = fov

        if intrinsics is not None:
            intrinsics_fov = 2 * np.rad2deg(np.arctan(0.5 * self._res[1] / intrinsics[0, 0]))
            if fov is not None:
                if abs(intrinsics_fov - fov) > 1e-4:
                    ezsim.raise_exception("The camera's intrinsic values and fov do not match.")
            else:
                self._fov = intrinsics_fov

        if self._rasterizer is not None:
            self._rasterizer.update_camera(self)
        if self._raytracer is not None:
            self._raytracer.update_camera(self)

    @ezsim.assert_built
    def start_recording(self):
        """
        Start recording on the camera. After recording is started, all the rgb images rendered by `camera.render()` will be stored, and saved to a video file when `camera.stop_recording()` is called.
        """
        self._in_recording = True

    @ezsim.assert_built
    def pause_recording(self):
        """
        Pause recording on the camera. After recording is paused, the rgb images rendered by `camera.render()` will not be stored. Recording can be resumed by calling `camera.start_recording()` again.
        """
        if not self._in_recording:
            ezsim.raise_exception("Recording not started.")
        self._in_recording = False


    #FIXME: add full support for preset "ultrafast, superfast, veryfast, faster, medium, slow, slower, veryslow, placebo"
    @ezsim.assert_built
    def stop_recording(self, save_to_filename=None, fps=60, preset:Literal['ultrafast','superfast','veryfast','faster']="ultrafast"):
        """
        Stop recording on the camera. Once this is called, all the rgb images stored so far will be saved to a video file. If `save_to_filename` is None, the video file will be saved with the name '{caller_file_name}_cam_{camera.idx}.mp4'.
        If `env_separate_rigid` in `VisOptions` is set to True, each environment will record and save a video separately. The filenames will be identified by the indices of the environments.

        Parameters
        ----------
        save_to_filename : str, optional
            Name of the output video file. If not provided, the name will be default to the name of the caller file, with camera idx, a timestamp and '.mp4' extension.
        fps : int, optional
            The frames per second of the video file.
        """

        if not self._in_recording:
            ezsim.raise_exception("Recording not started.")

        if save_to_filename is None:
            caller_file = inspect.stack()[-1].filename
            save_to_filename = (
                os.path.splitext(os.path.basename(caller_file))[0]
                + f'_cam_{self.idx}_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
            )

        if self._rgb_stacked:
            for env_idx in self._visualizer._context.rendered_envs_idx:
                env_imgs = [imgs[env_idx] for imgs in self._recorded_imgs]
                env_name, env_ext = os.path.splitext(save_to_filename)
                ezsim.tools.animate(env_imgs, f"{env_name}_{env_idx}{env_ext}", fps, preset=preset)
            #TODO: maybe add support for depth, segmentation and normal recording in stacked mode
        else:
            ezsim.tools.animate(self._recorded_imgs, save_to_filename, fps,preset=preset)
            ezsim.tools.animate(self._recorded_depths, f"{os.path.splitext(save_to_filename)[0]}_depth.mp4", fps, preset)
            ezsim.tools.animate(self._recorded_segmentations, f"{os.path.splitext(save_to_filename)[0]}_segmentation.mp4", fps, preset)
            ezsim.tools.animate(self._recorded_normals, f"{os.path.splitext(save_to_filename)[0]}_normal.mp4", fps, preset)

        self._recorded_imgs.clear()
        self._recorded_depths.clear()
        self._recorded_segmentations.clear()
        self._recorded_normals.clear()

        self._in_recording = False

    def _repr_brief(self):
        return f"{self._repr_type()}: idx: {self._idx}, pos: {self._pos}, lookat: {self._lookat}"

    @property
    def is_built(self):
        """Whether the camera is built."""
        return self._is_built

    @property
    def idx(self):
        """The integer index of the camera."""
        return self._idx

    @property
    def uid(self):
        """The unique ID of the camera"""
        return self._uid

    @property
    def model(self):
        """The camera model: `pinhole` or `thinlens`."""
        return self._model

    @property
    def res(self):
        """The resolution of the camera."""
        return self._res

    @property
    def fov(self):
        """The field of view of the camera."""
        return self._fov

    @property
    def aperture(self):
        """The aperture of the camera."""
        return self._aperture

    @property
    def focal_len(self):
        """The focal length for thinlens camera. Returns -1 for pinhole camera."""
        tan_half_fov = np.tan(np.deg2rad(self._fov / 2))
        if self.model == "thinlens":
            if self._res[0] > self._res[1]:
                projected_pixel_size = min(0.036 / self._res[0], 0.024 / self._res[1])
            else:
                projected_pixel_size = min(0.036 / self._res[1], 0.024 / self._res[0])
            image_dist = self._res[1] * projected_pixel_size / (2 * tan_half_fov)
            return 1.0 / (1.0 / image_dist + 1.0 / self._focus_dist)
        elif self.model == "pinhole":
            return self._res[0] / (2.0 * tan_half_fov)

    @property
    def focus_dist(self):
        """The focus distance of the camera."""
        return self._focus_dist

    @property
    def GUI(self):
        """Whether the camera will display the rendered images in a separate window."""
        return self._GUI

    @GUI.setter
    def GUI(self, value):
        self._GUI = value

    @property
    def spp(self):
        """Samples per pixel of the camera."""
        return self._spp

    @property
    def denoise(self):
        """Whether the camera will denoise the rendered image in raytracer."""
        return self._denoise

    @property
    def near(self):
        """The near plane of the camera."""
        return self._near

    @property
    def far(self):
        """The far plane of the camera."""
        return self._far

    @property
    def aspect_ratio(self):
        """The aspect ratio of the camera."""
        return self._aspect_ratio

    @property
    def pos(self):
        """The current position of the camera."""
        return np.array(self._pos)

    @property
    def lookat(self):
        """The current lookat point of the camera."""
        return np.array(self._lookat)

    @property
    def up(self):
        """The current up vector of the camera."""
        return np.array(self._up)

    @property
    def transform(self):
        """The current transform matrix of the camera."""
        return self._transform

    @property
    def extrinsics(self):
        """The current extrinsics matrix of the camera."""
        extrinsics = np.array(self.transform)
        extrinsics[:3, 1] *= -1
        extrinsics[:3, 2] *= -1
        return np.linalg.inv(extrinsics)

    @property
    def intrinsics(self):
        """The current intrinsics matrix of the camera."""
        # compute intrinsics using fov and resolution
        f = 0.5 * self._res[1] / np.tan(np.deg2rad(0.5 * self._fov))
        cx = 0.5 * self._res[0]
        cy = 0.5 * self._res[1]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
