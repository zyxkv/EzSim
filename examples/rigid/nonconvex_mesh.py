import argparse

import ezsim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(backend=ezsim.cpu if args.cpu else ezsim.gpu)

    ########################## create a scene ##########################
    scene = ezsim.Scene(
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    tank = scene.add_entity(
        ezsim.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 0),
        ),
        # vis_mode="collision",
    )
    ball = scene.add_entity(
        ezsim.morphs.Sphere(
            radius=0.1,
            pos=(0.0, 0.0, 1.0),
        ),
        # vis_mode="collision",
    )

    ########################## build ##########################
    scene.build()
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
