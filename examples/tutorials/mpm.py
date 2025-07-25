import ezsim

########################## init ##########################
ezsim.init()

########################## create a scene ##########################

scene = ezsim.Scene(
    sim_options=ezsim.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    mpm_options=ezsim.options.MPMOptions(
        lower_bound=(-0.5, -1.0, 0.0),
        upper_bound=(0.5, 1.0, 1),
    ),
    vis_options=ezsim.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    viewer_options=ezsim.options.ViewerOptions(
        camera_fov=30,
        res=(960, 640),
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=ezsim.morphs.Plane(),
)

obj_elastic = scene.add_entity(
    material=ezsim.materials.MPM.Elastic(),
    morph=ezsim.morphs.Box(
        pos=(0.0, -0.5, 0.25),
        size=(0.2, 0.2, 0.2),
    ),
    surface=ezsim.surfaces.Default(
        color=(1.0, 0.4, 0.4),
        vis_mode="visual",
    ),
)

obj_sand = scene.add_entity(
    material=ezsim.materials.MPM.Liquid(),
    morph=ezsim.morphs.Box(
        pos=(0.0, 0.0, 0.25),
        size=(0.3, 0.3, 0.3),
    ),
    surface=ezsim.surfaces.Default(
        color=(0.3, 0.3, 1.0),
        vis_mode="particle",
    ),
)

obj_plastic = scene.add_entity(
    material=ezsim.materials.MPM.ElastoPlastic(),
    morph=ezsim.morphs.Sphere(
        pos=(0.0, 0.5, 0.35),
        radius=0.1,
    ),
    surface=ezsim.surfaces.Default(
        color=(0.4, 1.0, 0.4),
        vis_mode="particle",
    ),
)


########################## build ##########################
scene.build()

horizon = 1000
for i in range(horizon):
    scene.step()
