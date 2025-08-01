import numpy as np
import pytest
import torch
import igl

import ezsim
from ezsim.utils.misc import tensor_to_array

from .utils import assert_allclose, get_hf_assets


@pytest.fixture(scope="session")
def fem_material():
    """Fixture for common FEM material properties"""
    return ezsim.materials.FEM.Muscle(
        E=3.0e4,
        nu=0.45,
        rho=1000.0,
        model="stable_neohookean",
    )


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_multiple_fem_entities(fem_material, show_viewer):
    """Test adding multiple FEM entities to the scene"""
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=ezsim.options.FEMOptions(
            damping=0.0,
        ),
        show_viewer=show_viewer,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material,
    )

    # Add second FEM entity
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=fem_material,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_interior_tetrahedralized_vertex(fem_material, show_viewer, box_obj_path, cube_verts_and_faces):
    """
    Test tetrahedralization of a FEM entity with a small maxvolume value that introduces
    internal vertices during tetrahedralization:
      1. Verify all surface vertices lie exactly on the original quad faces of the mesh.
      2. Ensure the visualizer's mesh triangles match the FEM entity's surface triangles.
    """
    verts, faces = cube_verts_and_faces

    scene = ezsim.Scene(
        show_viewer=show_viewer,
    )

    fem = scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            minratio=1.5,
            verbose=1,
            maxvolume=0.01,
        ),
        material=fem_material,
    )

    scene.build()

    state = fem.get_state()
    vertices = state.pos[0].cpu().numpy()
    surface_indices = np.unique(fem.surface_triangles)

    # Ensure there are interior vertices; this is a prerequisite for this test
    assert surface_indices.size < vertices.shape[0]

    # Verify each surface vertex lies on the original surface mesh
    def _point_on_surface(p, verts, faces, tol=1e-6):
        """Check if point p lies on any of the quad faces (as two triangles)."""
        for face in faces:
            # Convert 1-based face indices to 0-based
            idx = [i - 1 for i in face]
            # Extract vertices
            v0, v1, v2, v3 = [np.array(verts[i]) for i in idx]
            # Decompose quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
            for tri in ((v0, v1, v2), (v0, v2, v3)):
                a, b, c = tri
                # Compute normal for plane
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < tol:
                    continue
                # Check distance to plane
                distance = abs(np.dot(n / norm_n, p - a))
                if distance > tol:
                    continue
                # Barycentric coordinates
                v0v1 = b - a
                v0v2 = c - a
                v0p = p - a
                dot00 = np.dot(v0v2, v0v2)
                dot01 = np.dot(v0v2, v0v1)
                dot02 = np.dot(v0v2, v0p)
                dot11 = np.dot(v0v1, v0v1)
                dot12 = np.dot(v0v1, v0p)
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < tol:
                    continue
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                if u >= -tol and v >= -tol and (u + v) <= 1 + tol:
                    return True
        return False

    for idx in surface_indices:
        p = vertices[idx]
        assert _point_on_surface(
            p, verts, faces
        ), f"Surface vertex index {idx} with coordinate {p} does not lie on any original face"

    # Verify whether surface faces in the visualizer mesh matches the surface faces of the FEM entity
    rasterizer_context = scene.visualizer.context
    static_nodes = rasterizer_context.static_nodes
    fem_node_mesh = static_nodes[fem.uid].mesh

    (fem_node_primitive,) = fem_node_mesh.primitives
    fem_node_vertices = fem_node_primitive.positions
    fem_node_faces = fem_node_primitive.indices
    if fem_node_faces is None:
        fem_node_faces = np.arange(fem_node_vertices.shape[0]).reshape(-1, 3)

    def _make_triangle_set(verts, faces, tol=4):
        """
        Return a hashable, order-independent representation of a given set of triangle faces.

        Rounds each vertex coordinate to the given tolerance, sorts vertices within each triangle,
        and returns all triangles as a sorted tuple, eliminating any dependence on vertex or face order.
        """
        tri_set = set()
        for tri in faces:
            coords = [tuple(round(float(coord), tol) for coord in verts[i]) for i in tri]
            tri_set.add(tuple(sorted(coords)))
        return tuple(sorted(tri_set))

    # Triangles of FEM entity
    entity_tris = _make_triangle_set(vertices, fem.surface_triangles)

    # Triangles of visualizer
    viz_tris = _make_triangle_set(np.asarray(fem_node_vertices), np.asarray(fem_node_faces))

    assert entity_tris == viz_tris, (
        "FEM entity surface triangles and visualizer mesh triangles do not match.\n"
        f"Differences: {set(entity_tris) ^ set(viz_tris)}"
    )


@pytest.mark.parametrize("backend", [ezsim.cpu])
def test_maxvolume(fem_material, show_viewer, box_obj_path):
    """Test that imposing a maximum element volume constraint produces a finer mesh (i.e., more elements)."""
    scene = ezsim.Scene(
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    fem1 = scene.add_entity(
        morph=ezsim.morphs.Mesh(file=box_obj_path, nobisect=False, verbose=1),
        material=fem_material,
    )

    # Mesh with maximum element volume limited to 0.01
    fem2 = scene.add_entity(
        morph=ezsim.morphs.Mesh(file=box_obj_path, nobisect=False, maxvolume=0.01, verbose=1),
        material=fem_material,
    )

    assert len(fem1.elems) < len(fem2.elems), (
        f"Mesh with maxvolume=0.01 generated {len(fem2.elems)} elements; "
        f"expected more than {len(fem1.elems)} elements without a volume limit."
    )


@pytest.fixture(scope="session")
def fem_material_linear():
    """Fixture for common FEM linear material properties"""
    return ezsim.materials.FEM.Elastic()


def test_sphere_box_fall_implicit_fem_coupler(fem_material_linear, show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material_linear,
    )

    # Add second FEM entity
    scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=fem_material_linear,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(200):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        # The contact requires some penetration to generate enough contact force to cancel out gravity
        assert_allclose(
            min_pos_z, 0.0, atol=5e-2
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to 0.0."


def test_sphere_fall_implicit_fem_sap_coupler(fem_material_linear, show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=ezsim.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(0.5, -0.2, 0.5),
            radius=0.1,
        ),
        material=fem_material_linear,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        # The contact requires some penetration to generate enough contact force to cancel out gravity
        assert_allclose(
            min_pos_z, -1e-3, atol=1e-4
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to -1e-3."


@pytest.fixture(scope="session")
def fem_material_linear_corotated():
    """Fixture for common FEM linear material properties"""
    return ezsim.materials.FEM.Elastic(model="linear_corotated")


def test_linear_corotated_sphere_fall_implicit_fem_sap_coupler(fem_material_linear_corotated, show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        # Not using default fem_options to make it faster, linear material only need one iteration without linesearch
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=ezsim.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(0.5, -0.2, 0.5),
            radius=0.1,
        ),
        material=fem_material_linear_corotated,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        pos = tensor_to_array(state.pos.reshape(-1, 3))
        min_pos_z = np.min(pos[..., 2])
        # The contact requires some penetration to generate enough contact force to cancel out gravity
        assert_allclose(
            min_pos_z, -1e-3, atol=1e-4
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to -1e-3."
        BV, BF = igl.bounding_box(pos)
        x_scale = BV[0, 0] - BV[-1, 0]
        y_scale = BV[0, 1] - BV[-1, 1]
        z_scale = BV[0, 2] - BV[-1, 2]
        assert_allclose(x_scale, 0.2, atol=1e-3), f"Entity {entity.uid} X scale {x_scale} is not close to 0.2."
        assert_allclose(y_scale, 0.2, atol=1e-3), f"Entity {entity.uid} Y scale {y_scale} is not close to 0.2."
        # The Z scale is expected to be more squashed due to gravity
        assert_allclose(z_scale, 0.2, atol=2e-3), f"Entity {entity.uid} Z scale {z_scale} is not close to 0.2."


@pytest.fixture(scope="session")
def fem_material_linear_corotated_soft():
    """Fixture for common FEM linear material properties"""
    return ezsim.materials.FEM.Elastic(model="linear_corotated", E=1.0e5, nu=0.4)


def test_fem_sphere_box_self(fem_material_linear_corotated, fem_material_linear_corotated_soft, show_viewer):
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=ezsim.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=ezsim.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=fem_material_linear_corotated,
    )

    # Add second FEM entity
    scale = 0.1
    asset_path = get_hf_assets(pattern="meshes/cube8.obj")
    scene.add_entity(
        morph=ezsim.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=scale,
            pos=(0.0, 0.0, scale * 4.0),
        ),
        material=fem_material_linear_corotated,
    )

    # Build the scene
    scene.build()
    # Run simulation
    for _ in range(200):
        scene.step()

    depths = [-1e-3, -2e-5]
    atols = [2e-4, 4e-6]
    for i, entity in enumerate(scene.entities):
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        # The contact requires some penetration to generate enough contact force to cancel out gravity
        assert_allclose(
            min_pos_z, depths[i], atol=atols[i]
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to {depths[i]}."


def test_box_hard_vertex_constraint(show_viewer):
    """
    Test if a box with hard vertex constraints has those vertices fixed,
    and that updating and removing constraints works correctly.
    """
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=ezsim.materials.FEM.Elastic(),
    )
    verts_idx = [0, 3]
    initial_target_poss = box.init_positions[verts_idx]

    scene.build(n_envs=2)

    if show_viewer:
        scene.draw_debug_spheres(poss=initial_target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    box.set_vertex_constraints(verts_idx=verts_idx, target_poss=initial_target_poss)

    for _ in range(100):
        scene.step()

    positions = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions, initial_target_poss, tol=0.0
    ), "Vertices should stay at initial target positions with hard constraints"
    new_target_poss = initial_target_poss + ezsim.tensor(
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    )
    box.update_constraint_targets(verts_idx=verts_idx, target_poss=new_target_poss)

    for _ in range(100):
        scene.step()

    positions_after_update = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions_after_update, new_target_poss, tol=0.0
    ), "Vertices should be at new target positions after updating constraints"

    box.remove_vertex_constraints()

    for _ in range(100):
        scene.step()

    positions_after_removal = box.get_state().pos[0][verts_idx]

    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            positions_after_removal, new_target_poss, tol=1e-3
        ), "Vertices should have moved after removing constraints"


def test_box_soft_vertex_constraint(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    scene = ezsim.Scene(
        sim_options=ezsim.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        fem_options=ezsim.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=ezsim.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=ezsim.materials.FEM.Elastic(),
    )
    verts_idx = [0, 1]
    target_poss = box.init_positions[verts_idx]

    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(poss=target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    box.set_vertex_constraints(
        verts_idx=verts_idx,
        target_poss=target_poss,
        is_soft_constraint=True,
        stiffness=2.0e5,
    )
    box.set_velocity(ezsim.tensor([1.0, 1.0, 1.0]) * 1e-2)

    for _ in range(500):
        scene.step()

    positions = box.get_state().pos[0][verts_idx]

    assert_allclose(
        positions, target_poss, tol=5e-5
    ), "Vertices should be near target positions with strong soft constraints"
