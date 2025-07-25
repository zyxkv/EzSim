import ezsim

########################## init ##########################
ezsim.init()

########################## create a scene ##########################

scene = ezsim.Scene(
    sim_options=ezsim.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    sph_options=ezsim.options.SPHOptions(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1),
        particle_size=0.01,
    ),
    vis_options=ezsim.options.VisOptions(
        visualize_sph_boundary=True,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=ezsim.morphs.Plane(),
)

liquid = scene.add_entity(
    # viscous liquid
    # material=ezsim.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=ezsim.materials.SPH.Liquid(),
    morph=ezsim.morphs.Box(
        pos=(0.0, 0.0, 0.65),
        size=(0.4, 0.4, 0.4),
    ),
    surface=ezsim.surfaces.Default(
        color=(0.4, 0.8, 1.0),
        vis_mode="particle",
    ),
)

########################## build ##########################
scene.build()

horizon = 1000
for i in range(horizon):
    scene.step()

# get particle positions
particles = liquid.get_particles()
