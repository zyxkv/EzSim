import torch

import ezsim

########################## init ##########################
ezsim.init(backend=ezsim.gpu)

########################## create a scene ##########################
scene = ezsim.Scene(
    show_viewer=False,
    viewer_options=ezsim.options.ViewerOptions(
        camera_pos=(3.5, -1.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    rigid_options=ezsim.options.RigidOptions(
        dt=0.01,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    ezsim.morphs.Plane(),
)

franka = scene.add_entity(
    ezsim.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

########################## build ##########################

# create 20 parallel environments
B = 20
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# control all the robots
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=ezsim.device), (B, 1)),
)

# control only 3 environments: 1, 5, and 7. (You also need to comment out the function call above)
# franka.control_dofs_position(
#     position = torch.zeros(3, 9, device=ezsim.device),
#     envs_idx = torch.tensor([1, 5, 7], device=ezsim.device),
# )
for i in range(1000):
    scene.step()
