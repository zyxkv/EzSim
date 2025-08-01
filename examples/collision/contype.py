"""
NOTE: contype and conaffinity are 32-bit integer bitmasks used for contact filtering of contact pairs.
When the contype of one geom and the conaffinity of the other geom share a common bit set to 1, two geoms can collide.
Plane:      contype=0xFFFF, conaffinity=0xFFFF (1111 1111 1111 1111)
Red Cube:   contype=1, conaffinity=1 (0001) -> collide with Plane and Blue Cube
Green Cube: contype=2, conaffinity=2 (0010) -> collide with Plane and Blue Cube
Blue Cube:  contype=3, conaffinity=3 (0011) -> collide with Plane, Red Cube, and Green Cube
Dragon:     contype=4, conaffinity=4 (0100) -> collide with Plane only
"""

import argparse

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ezsim.init()

    scene = ezsim.Scene(
        viewer_options=ezsim.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(ezsim.morphs.Plane())

    scene.add_entity(
        ezsim.morphs.Box(
            pos=(0.025, 0, 0.5),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=1,
            conaffinity=1,
        ),
        surface=ezsim.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.add_entity(
        ezsim.morphs.Box(
            pos=(-0.025, 0, 1.0),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=2,
            conaffinity=2,
        ),
        surface=ezsim.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    scene.add_entity(
        ezsim.morphs.Box(
            pos=(0.0, 0, 1.5),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=3,
            conaffinity=3,
        ),
        surface=ezsim.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.004,
            euler=(0, 0, 90),
            pos=(-0.1, 0.0, 1.0),
            contype=4,
            conaffinity=4,
        ),
    )

    scene.build()

    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()