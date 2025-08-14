import ezsim 
from ezsim.utils.image_exporter import FrameImageExporter


def main():
    ########################## init ##########################
    ezsim.init(
        seed=0, 
        precision="32", 
        logging_level=None,
        log_time = False,
    )

    ########################## create a scene ##########################
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(),
        viewer_options=ezsim.options.ViewerOptions(
            res=(1600, 900),
            camera_pos=(8.5, 0.0, 4.5),
            camera_lookat=(3.0, 0.0, 0.5),
            camera_fov=50,
        ),
        rigid_options=ezsim.options.RigidOptions(
            enable_collision=False, 
            gravity=(0, 0, -9.81)
        ),
        renderer=ezsim.options.renderers.BatchRenderer(
            use_rasterizer=True,
        ),
        show_viewer=False,  # Set to True if you want to see the viewer
    )

    ########################## materials ##########################

    ########################## entities ##########################
    # floor
    plane = scene.add_entity(
        morph=ezsim.morphs.Plane(pos=(0.0, 0.0, -0.5),),
        surface=ezsim.surfaces.Aluminium(ior=10.0,),
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
        res=(640, 480),
        pos=(5, 0.0, 4),
        lookat=(3.0, 0.0, 0.5),
        fov=70,
        GUI=False,
        spp=512,
    )
    scene.add_light(
        pos=[0.0, 0.0, 4.5],
        dir=[-1.0, -1.0, -1.0],
        directional=1,
        castshadow=1,
        cutoff=45.0,
        intensity=1.0,
    )
    scene.build()

    ########################## forward + backward twice ##########################
    scene.reset()
    horizon = 10

    # Create an image exporter
    output_dir = "img_output/texture_demo"
    exporter = FrameImageExporter(output_dir)
    try:
        for i in range(horizon):
            scene.step()
            rgb, depth, normal, segmentation = scene.render_all_cameras(depth=True,normal=True,segmentation=True)
            exporter.export_frame_all_cameras(i, rgb=rgb, depth=depth, normal=normal, segmentation=segmentation)
    except KeyboardInterrupt:
        ezsim.logger.info("Simulation interrupted, exiting.")
    finally:
        ezsim.logger.info("Simulation finished.")

if __name__ == "__main__":
    main()
