import argparse

import numpy as np

import ezsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    ezsim.init(backend=ezsim.gpu)

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
            dt=0.01,
            gravity=(0, 0, 0),
        ),
        vis_options=ezsim.options.VisOptions(
            show_link_frame=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    duck = scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0, 0, 0.0),
        ),
    )
    ########################## build ##########################
    scene.build()

    dofs_idx = duck.base_joint.dofs_idx

    duck.set_dofs_kv(
        np.array([1, 1, 1, 1, 1, 1]) * 50.0,
        dofs_idx,
    )
    pos = duck.get_dofs_position()
    pos[-1] = 1.0  # rotate around intrinsic z axis
    duck.control_dofs_position(
        pos,
        dofs_idx,
    )
    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
