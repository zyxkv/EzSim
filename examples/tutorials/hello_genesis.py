import ezsim

ezsim.init(backend=ezsim.cpu)

scene = ezsim.Scene()

plane = scene.add_entity(
    ezsim.morphs.Plane(),
)
franka = scene.add_entity(
    # ezsim.morphs.URDF(
    #     file='urdf/panda_bullet/panda.urdf',
    #     fixed=True,
    # ),
    ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()
for i in range(1000):
    scene.step()
