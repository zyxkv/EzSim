import ezsim

########################## init ##########################
ezsim.init()

########################## create a scene ##########################

scene = ezsim.Scene(
    sim_options=ezsim.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    viewer_options=ezsim.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=ezsim.morphs.Plane(),
)

cloth_1 = scene.add_entity(
    material=ezsim.materials.PBD.Cloth(),
    morph=ezsim.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 0.5),
        euler=(0.0, 0, 0.0),
    ),
    surface=ezsim.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode="visual",
    ),
)

cloth_2 = scene.add_entity(
    material=ezsim.materials.PBD.Cloth(),
    morph=ezsim.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 1.0),
        euler=(0.0, 0, 0.0),
    ),
    surface=ezsim.surfaces.Default(
        color=(0.8, 0.4, 0.2, 1.0),
        vis_mode="particle",
    ),
)

########################## build ##########################
scene.build()

cloth_1.fix_particle(cloth_1.find_closest_particle((-1, -1, 1.0)), 0)
cloth_1.fix_particle(cloth_1.find_closest_particle((1, 1, 1.0)), 0)
cloth_1.fix_particle(cloth_1.find_closest_particle((-1, 1, 1.0)), 0)
cloth_1.fix_particle(cloth_1.find_closest_particle((1, -1, 1.0)), 0)

cloth_2.fix_particle(cloth_2.find_closest_particle((-1, -1, 1.0)), 0)

horizon = 1000
for i in range(horizon):
    scene.step()
