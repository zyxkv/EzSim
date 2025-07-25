import argparse

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(backend=ezsim.gpu)

    scene = ezsim.Scene(
        show_viewer=False,
        vis_options=ezsim.options.VisOptions(
            plane_reflection=False,
        ),
        rigid_options=ezsim.options.RigidOptions(
            dt=0.01,
            # enable_collision=True,
            # enable_joint_limit=True,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    cube = scene.add_entity(
        ezsim.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0, 0, 0),
        ),
    )
    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(640, 480),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
    )
    ########################## build ##########################
    scene.build()

    from time import time

    t = time()
    for i in range(2000):
        cam_0.render(rgb=True, depth=True)
    print(2000 / (time() - t), "FPS")
    exit()


if __name__ == "__main__":
    main()
