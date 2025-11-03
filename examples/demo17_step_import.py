# -----------------------------------------------------------------------------
# STEP File import
#
# In this example we will look at how the STEP file import module can be used.
# Please read the following CAREFULLY
# -----------------------------------------------------------------------------

############################################################
#                         IMPORTANT                        #
############################################################
"""
This example file demonstrates how to import and use STEP files in EMerge.
Try toggling the if-statements on/off to explore behavior and meshing issues.
If the simulation halts, check if any display windows are hidden behind others.
"""

import emerge as em
from emerge.plot import plot_sp
from pathlib import Path

mm = 0.001  # unit helper: 1 mm in meters

margin = 5*mm
model = em.Simulation('STEP_Board_demo')  # create simulation container
model.check_version('1.2.1') # Checks version compatibility.

# STEPItems is a wrapper that loads geometries from a STEP file.
# It can contain multiple solids, surfaces, edges, and points.
# These are accessible via .volumes, .faces, .edges, and .points.
step = em.geo.STEPItems('board', str(Path(__file__).parent / 'Connector_MicrostripBoard_Solder.step'), 0.001)

# The imported STEP geometry can include multiple disconnected volumes.
# Visual inspection helps identify which volume represents which component.
# You can use m.generate_mesh() and m.view() to preview.
# Large cylindrical or thin features may cause meshing errors if element size is too coarse.

# SET TRUE TO LET IT CRASH
# Below block intentionally crashes to demonstrate Gmsh limitations on narrow cylinders.
if False:
    model.commit_geometry()
    model.mw.set_frequency(1e9)
    model.generate_mesh()
    model.view()

# SET TRUE TO MESH AND VIEW THE GEOMETRY
# This version refines the mesh to avoid intersection errors and label the volumes.
if False:
    # Inspect labeled objects "board_Obj0", "board_Obj1", etc. to identify geometry parts.
    for vol in step.volumes:
        vol.max_meshsize = 1e-3  # smaller element size improves meshing stability
    model.mw.set_frequency(1e9)
    model.generate_mesh()
    model.view(labels=True)

# After identifying each volume from visualization, assign materials accordingly.
trace, diel, ground, conn_diel, conn_block, conn_pin = step.volumes

trace.set_material(em.lib.COPPER)
diel.set_material(em.lib.DIEL_RO4350B)
ground.set_material(em.lib.COPPER)
conn_diel.set_material(em.lib.DIEL_TEFLON)
conn_block.set_material(em.lib.COPPER)
conn_pin.set_material(em.lib.COPPER)

volumes = (trace, diel, ground, conn_diel, conn_block, conn_pin)

# Define a lumped port surface using a small plate geometry at a known connector location.
# These coordinates were found interactively using m.view().
line_w = 1.1*mm
th = (1.01*mm - 0.5*mm)
lumped_port_face = em.geo.Plate((-line_w/2, 35*mm, -1.01*mm), (line_w, 0, 0), (0, 0, th))

# Create an enclosing air box around the full geometry, with given XYZ margins.
air = step.enclose(x_margins=(margin, margin), y_margins=(-1*mm, margin), z_margins=(margin, margin))

# Subtract solid geometries from the air domain to prevent overlapping volumes in meshing.
for vol in volumes:
    air = em.geo.subtract(air, vol, remove_tool=False)

# Commit final geometry setup to the solver.
model.commit_geometry()

# Set local mesh size limits to avoid "PLC Error" during tetrahedralization.
conn_pin.max_meshsize = 0.3*mm
conn_diel.max_meshsize = 0.6*mm
conn_block.max_meshsize = 1*mm

# Define frequency sweep and basic mesh refinement for conductor boundaries.
model.mw.set_frequency_range(0.1e9, 10e9, 41)
model.mesher.set_boundary_size(trace, 2*mm)
model.generate_mesh()

# View mesh to verify structure and ensure no errors before solving.
model.view(plot_mesh=True, volume_mesh=False)

# Identify boundary faces using automatic directional tags: -x, +x, -y, +y, -z, +z.
# Here, the coaxial connector input is on the -y face.
coax_port_face = conn_diel.face('-y')
# Radiation boundary is applied to all outer box faces except the coax connector.
radiation_face = air.faces('-x', '+x', '-z', '+z', '+y')
model.view(selections=[coax_port_face,])

# Define boundary conditions:
#   Port 1: coaxial modal port on the connector side.
#   Radiation: absorbing boundary on air box.
#   Port 2: lumped port representing microstrip connection.
model.mw.bc.ModalPort(coax_port_face, 1, modetype='TEM')
model.mw.bc.AbsorbingBoundary(radiation_face)
model.mw.bc.LumpedPort(lumped_port_face, 2, line_w, th, (0, 0, 1))

# Run the frequency-domain sweep with parallel processing enabled.
data = model.mw.run_sweep(True, 4, frequency_groups=8)

# Extract structured S-parameter data.
g = data.scalar.grid

# Plot S11 and S21 responses to observe connector-to-trace coupling.
plot_sp(g.freq, [g.S(1,1), g.S(2,1)], labels=['S11', 'S21'])

# Extract field data at 2.25 GHz for near-field and far-field visualization.
field = data.field.find(freq=2.25e9)

# Compute far-field pattern on the defined radiation boundary.
ff = field.farfield_3d(radiation_face)

# Setup visualization with geometry, field cutplane, and far-field surface overlay.
d = model.display
d.add_objects(*model.all_geos())
d.add_surf(*field.cutplane(0.5*mm, x=0).scalar('Ez', 'real'), symmetrize=True)
d.add_surf(*ff.surfplot('normE', rmax=10*mm, offset=(0, 25*mm, 0)))
d.show()
