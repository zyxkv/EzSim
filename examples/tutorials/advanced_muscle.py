import numpy as np
import ezsim


########################## init ##########################
ezsim.init(seed=0, precision="32", logging_level="debug")

########################## create a scene ##########################
dt = 5e-4
scene = ezsim.Scene(
    sim_options=ezsim.options.SimOptions(
        substeps=10,
        gravity=(0, 0, 0),
    ),
    viewer_options=ezsim.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    mpm_options=ezsim.options.MPMOptions(
        dt=dt,
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
    ),
    fem_options=ezsim.options.FEMOptions(
        dt=dt,
        damping=45.0,
    ),
    vis_options=ezsim.options.VisOptions(
        show_world_frame=False,
    ),
    show_viewer=True,
)

########################## entities ##########################
scene.add_entity(morph=ezsim.morphs.Plane())

E, nu = 3.0e4, 0.45
rho = 1000.0

robot_mpm = scene.add_entity(
    morph=ezsim.morphs.Sphere(
        pos=(0.5, 0.2, 0.3),
        radius=0.1,
    ),
    material=ezsim.materials.MPM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model="neohooken",
    ),
)

robot_fem = scene.add_entity(
    morph=ezsim.morphs.Sphere(
        pos=(0.5, -0.2, 0.3),
        radius=0.1,
    ),
    material=ezsim.materials.FEM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model="stable_neohookean",
    ),
)

########################## build ##########################
scene.build(n_envs=5)

########################## run ##########################
scene.reset()
for i in range(1000):
    actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])

    robot_mpm.set_actuation(actu)
    robot_fem.set_actuation(actu)
    scene.step()
