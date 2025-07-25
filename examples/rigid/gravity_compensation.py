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
            camera_lookat=(0.0, 1.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        ezsim.morphs.Plane(),
    )
    scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0)),
    )
    scene.add_entity(
        ezsim.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 1.0, 0.0),
        ),
        material=ezsim.materials.Rigid(gravity_compensation=0.5),
    )
    scene.add_entity(
        ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 2.0, 0.0)),
        material=ezsim.materials.Rigid(gravity_compensation=1.0),
    )

    scene.build()
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
