import torch

import ezsim


def main():
    w = 640
    h = 480
    fps = 30
    video_len = 2  # seconds

    ########################## init ##########################
    ezsim.init(
        seed                = 4320,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        logging_level       = None, #'warning',
        backend             = ezsim.cuda,
        theme               = 'dark',
        logger_verbose_time = False
    )
    ########################## create a scene ##########################
    scene = ezsim.Scene(
        show_viewer=False,
        rigid_options=ezsim.options.RigidOptions(enable_collision=False, gravity=(0, 0, 0)),
        viewer_options=ezsim.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(8.5, 0.0, 4.5),
            camera_lookat=(3.0, 0.0, 0.5),
            camera_fov=50,
        ),
        renderer=ezsim.renderers.RayTracer(  # type: ignore
            env_surface=ezsim.surfaces.Emission(
                emissive_texture=ezsim.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        ),
    )

    ########################## materials ##########################

    ########################## entities ##########################
    # floor
    plane = scene.add_entity(
        morph=ezsim.morphs.Plane(
            pos=(0.0, 0.0, -0.5),
        ),
        surface=ezsim.surfaces.Aluminium(
            ior=10.0,
        ),
    )

    # user specified external color texture
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -3, 0.0),
        ),
        surface=ezsim.surfaces.Rough(
            diffuse_texture=ezsim.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    # user specified color (using color shortcut)
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -1.8, 0.0),
        ),
        surface=ezsim.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # smooth shortcut
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -0.6, 0.0),
        ),
        surface=ezsim.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    # Iron
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 0.6, 0.0),
        ),
        surface=ezsim.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Gold
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 1.8, 0.0),
        ),
        surface=ezsim.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Glass
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 3.0, 0.0),
        ),
        surface=ezsim.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Opacity
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(2.0, -3, 0.0),
        ),
        surface=ezsim.surfaces.Smooth(color=(1.0, 1.0, 1.0, 0.5)),
    )
    # asset's own attributes
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -2.3, 0.0),
        ),
    )
    # override asset's attributes
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -1.0, 0.0),
        ),
        surface=ezsim.surfaces.Rough(
            diffuse_texture=ezsim.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(w, h),
        pos=(8.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.7),
        fov=60,
        GUI=False,
        spp=512,
    )
    scene.build()
    cam_0.start_recording()

    ########################## forward + backward twice ##########################
    scene.reset()

    for i in range(video_len*fps):
        scene.step()
        cam_0.render()
    cam_0.stop_recording(
        save_to_filename=f'demo_raytrace_{w}x{h}_{fps}fps.mp4', 
        fps=fps)



if __name__ == "__main__":
    main()
