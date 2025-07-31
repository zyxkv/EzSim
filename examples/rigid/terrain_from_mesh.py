import argparse
import os
import time
import ezsim
import numpy as np
from ezsim.utils.terrain import mesh_to_heightfield
from ezsim.sensors import SensorDataRecorder, VideoFileWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--w", type=int, default=640, help="Camera width")
    parser.add_argument("--h", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_len", type=int, default=10, help="Video length in seconds")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(backend=ezsim.cpu if args.cpu else ezsim.gpu)

    ########################## create a scene ##########################
    scene = ezsim.Scene(
        # viewer_options=ezsim.options.ViewerOptions(
        #     camera_pos=(0, -50, 0),
        #     camera_lookat=(0, 0, 0),
        # ),
        sim_options=ezsim.options.SimOptions(dt=args.dt),
        rigid_options=ezsim.options.RigidOptions(
            dt=args.dt,
        ),
        profiling_options=ezsim.options.ProfilingOptions(
            show_FPS=False
        ),
        vis_options=ezsim.options.VisOptions(show_world_frame=True),
        show_viewer=args.vis,
    )
    ########################## add entities ##########################

    # 使用更小的horizontal_scale来获得更高分辨率的heightfield
    
    ezsim_root = os.path.dirname(os.path.abspath(ezsim.__file__))
    # path_terrain = os.path.join(ezsim_root, "assets", "meshes", "terrain_45.obj")
    path_terrain = os.path.join(ezsim_root, "..", "ext_assets", "terrain.glb")
    
    # 使用更高的oversample来捕获更多细节，并使用不同的spacing处理窄mesh
    # 对于Y方向较窄的mesh，使用更小的spacing
    spacing_x = 0.1  # X方向spacing，减小以获得更多网格点
    spacing_y = 0.02  # Y方向使用更小的spacing，因为mesh在Y方向很窄
    horizontal_scale = max(spacing_x,spacing_y)  # 从2.0改为0.5，获得更密集的网格
    print("=== Loading terrain mesh ===")
    hf_terrain, xs, ys = mesh_to_heightfield(
        path_terrain, spacing=(spacing_x, spacing_y), 
        oversample=3, 
        up_axis='y' if path_terrain.endswith('.glb') else 'z'
    )

    print("=== Heightfield Analysis ===")
    print(f"Heightfield shape: {hf_terrain.shape}")
    print(f"Height range: min={np.min(hf_terrain):.3f}, max={np.max(hf_terrain):.3f}")
    print(f"X range: {np.min(xs):.2f} to {np.max(xs):.2f} (span: {np.max(xs) - np.min(xs):.2f})")
    print(f"Y range: {np.min(ys):.2f} to {np.max(ys):.2f} (span: {np.max(ys) - np.min(ys):.2f})")
    print(f"Z range: {np.min(hf_terrain):.2f} to {np.max(hf_terrain):.2f} (span: {np.max(hf_terrain) - np.min(hf_terrain):.2f})")
    print(f"NaN values: {np.sum(np.isnan(hf_terrain))} out of {hf_terrain.size}")
    
    # 检查是否还有NaN值，如果有就用最小高度填充
    if np.any(np.isnan(hf_terrain)):
        min_height = np.nanmin(hf_terrain)
        hf_terrain = np.where(np.isnan(hf_terrain), min_height, hf_terrain)
        print(f"Warning: Filled {np.sum(np.isnan(hf_terrain))} NaN values with {min_height:.3f}")

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.min(xs), np.min(ys), 0])

    terrain_heightfield = scene.add_entity(
        morph=ezsim.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )

    # 调整球的位置到地形范围内，并放在地形上方
    ball_x = (np.min(xs) + np.max(xs)) / 2  # X方向中心
    ball_y = (np.min(ys) + np.max(ys)) / 2  # Y方向中心  
    ball_z = np.max(hf_terrain) + 3  # 在最高点上方3单位
    
    ball = scene.add_entity(
        ezsim.morphs.Sphere(
            pos=(ball_x, ball_y, ball_z),
            radius=0.01,
        ),
        vis_mode="collision",
    )
    
    print(f"=== Ball positioned at: ({ball_x:.2f}, {ball_y:.2f}, {ball_z:.2f}) ===")
    ########################## add sensors ##########################
    data_recorder = SensorDataRecorder(step_dt=args.dt)
    
    # 调整相机位置以更好地观察地形
    cam_x = (np.min(xs) + np.max(xs)) / 2  # X方向中心
    cam_y = np.min(ys) - 50  # 在Y方向前方
    cam_z = np.max(hf_terrain) + 20  # 在地形上方
    
    cam = scene.add_camera(
        res=(args.w, args.h),
        pos=(cam_x, cam_y, cam_z),
        lookat=(cam_x, 
                (np.min(ys) + np.max(ys)) / 2, 
                np.mean(hf_terrain)
               ),
        fov=40,
        GUI=False,
    )
    
    print(f"=== Camera positioned at: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) ===")

    # we can also record the camera video using data_recorder
    data_recorder.add_sensor(cam, VideoFileWriter(filename="terrain_from_mesh.mp4"))
    ########################## build ##########################

    scene.build()
    data_recorder.start_recording()
    try:
        for i in range(args.video_len*args.fps):
            scene.step()
            # cam.set_pose(
            #     pos    = (5 * np.sin(i / args.fps), 
            #               5 * np.cos(i / args.fps), 
            #               cam_z),
            #     lookat = (cam_x, (np.min(ys) + np.max(ys)) / 2, np.mean(hf_terrain)),
            # )
            data_recorder.step()
            time.sleep(1/args.fps)
    except KeyboardInterrupt:
        ezsim.logger.info("Simulation interrupted, exiting.")
    finally:
        ezsim.logger.info("Simulation finished.")

        data_recorder.stop_recording()


if __name__ == "__main__":
    main()
