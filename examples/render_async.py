import threading

import ezsim


def run_sim(scene):
    for _ in range(200):
        scene.step(refresh_visualizer=False)


def main():
    ########################## init ##########################
    ezsim.init()

    ########################## create a scene ##########################

    scene = ezsim.Scene(
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            run_in_thread=False,
        ),
        show_viewer=True,
        show_FPS=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(ezsim.morphs.Plane())
    r0 = scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    ########################## build ##########################
    scene.build()

    threading.Thread(target=run_sim, args=(scene,)).start()
    scene.viewer.run()


if __name__ == "__main__":
    main()
