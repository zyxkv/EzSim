import argparse
import os

import ezsim
import numpy as np
from ezsim.utils.terrain import mesh_to_heightfield


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
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=args.vis,
    )

    horizontal_scale = 2.0
    ezsim_root = os.path.dirname(os.path.abspath(ezsim.__file__))
    path_terrain = os.path.join(ezsim_root, "assets", "meshes", "terrain_45.obj")
    hf_terrain, xs, ys = mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)
    print("hf_terrain", path_terrain, hf_terrain.shape, np.max(hf_terrain))

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    terrain_heightfield = scene.add_entity(
        morph=ezsim.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )

    ball = scene.add_entity(
        ezsim.morphs.Sphere(
            pos=(10, 15, 10),
            radius=1,
        ),
        vis_mode="collision",
    )

    scene.build()

    for i in range(2000):
        scene.step()


if __name__ == "__main__":
    main()
