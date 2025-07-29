import time
import argparse
import numpy as np
import ezsim
from ezsim.sensors import SensorDataRecorder, VideoFileWriter

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=2.0, help="Number of seconds to simulate")
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Number of substeps",
    )
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(
        backend=ezsim.gpu,
        logger_verbose_time=True
    )

    ########################## create a scene ##########################
    viewer_options = ezsim.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=200,
    )

    scene = ezsim.Scene(
        viewer_options=viewer_options,
        rigid_options=ezsim.options.RigidOptions(
            dt=args.dt,
            # gravity=(0, 0, 0),
        ),
        vis_options=ezsim.options.VisOptions(
            show_link_frame=False,
            show_world_frame=False,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        morph=ezsim.morphs.Plane(),
    )
    duck = scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0, 0, 1.0),
        ),
    )
    ########################## add sensors ##########################
    data_recorder = SensorDataRecorder(step_dt=args.dt)
    # Add camera for visualization
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(0, -3.5, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=70,
        GUI=args.vis,
    )
    # we can also record the camera video using data_recorder
    data_recorder.add_sensor(cam, VideoFileWriter(filename="acc_duck.mp4"))
    ########################## build ##########################
    scene.build()
    dofs_idx = duck.base_joint.dofs_idx

    duck.set_dofs_kv(
        np.array([1, 1, 1, 1, 1, 1]) * 50.0,
        dofs_idx,
    )
    pos = duck.get_dofs_position()
    pos[-1] = -1  # rotate around intrinsic z axis
    # duck.control_dofs_position(
    #     pos,
    #     dofs_idx,
    # )
    data_recorder.start_recording()
    try:
        steps = int(args.seconds / args.dt)
        for _ in tqdm(range(steps), total=steps):
            scene.step()

            # visualize
            # links_acc = duck.get_links_acc()
            links_acc = duck.get_links_accelerometer_data()
            links_pos = duck.get_links_pos()
            
            for i in range(links_acc.shape[0]):
                link_pos = links_pos[i]
                link_acc = links_acc[i]
                link_acc *= 1000 if link_acc.norm() < 0.001 else 1/link_acc.norm() # scale for better visualization
                scene.draw_debug_arrow(
                    pos=link_pos.tolist(),
                    vec=link_acc.tolist(),
                )
            # print(link_acc, link_acc.norm())
            data_recorder.step()
            scene.clear_debug_objects()
            time.sleep(0.03)
    except KeyboardInterrupt:
        ezsim.logger.info("Simulation interrupted, exiting.")
    finally:
        ezsim.logger.info("Simulation finished.")

        data_recorder.stop_recording()

        # print("Max force recorded:", max_observed_force_magnitude)


if __name__ == "__main__":
    main()
