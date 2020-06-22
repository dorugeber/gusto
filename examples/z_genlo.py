from firedrake import (CubedSphereMesh, FunctionSpace, MeshHierarchy, inject,
                       restrict, prolong, SpatialCoordinate, Function, File,
                       TransferManager, pi, sqrt, Min, FiniteElement)
from gusto import latlon_coords
import numpy as np

baseref = 2
ref_hi = 5
ref_lo = 4

R = 6371220.0

assert ref_hi >= ref_lo >= baseref

mesh0 = CubedSphereMesh(radius=R, refinement_level=baseref)
hierarchy = MeshHierarchy(mesh0, ref_hi-baseref)
mesh_hi = hierarchy[ref_hi - baseref]
mesh_lo = hierarchy[ref_lo - baseref]

for msh in (mesh_hi, mesh_lo):
    msh.coordinates.dat.data[:] *= (R / np.linalg.norm(msh.coordinates.dat.data, axis=1)).reshape(-1, 1)

for msh in (mesh_hi, mesh_lo):
    x = SpatialCoordinate(msh)
    msh.init_cell_orientations(x)

cell = mesh0.ufl_cell().cellname()
V_elt = FiniteElement("RTCF", cell, 2, variant="equispaced")
W_elt = FiniteElement("DG", cell, 1, variant="equispaced")

V_hi = FunctionSpace(mesh_hi, V_elt)
W_hi = FunctionSpace(mesh_hi, W_elt)

V_lo = FunctionSpace(mesh_lo, V_elt)
W_lo = FunctionSpace(mesh_lo, W_elt)

u0_hi = Function(V_hi)
D0_hi = Function(W_hi)

# topography mess
bexprs = []
for msh in (mesh_hi, mesh_lo):
    x = SpatialCoordinate(msh)
    theta, lamda = latlon_coords(msh)
    Rsq = R**2
    R0 = pi/9.0
    R0sq = R0**2
    lamda_c = -pi/2.0
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = Min(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    bexprs.append(bexpr)

b_hi = Function(W_hi).interpolate(bexprs[0])
b_lo = Function(W_lo).interpolate(bexprs[1])

# end topography

### Do ICs

u0_lo = Function(V_lo)
D0_lo = Function(W_lo)
tm = TransferManager()
D0pb_hi = Function(W_hi)
D0pb_lo = Function(W_lo)

u0_hi.dat.data[:] = np.load("day50-u.npy")
D0_hi.dat.data[:] = np.load("day50-D.npy")

D0pb_hi.assign(D0_hi + b_hi)
tm.inject(u0_hi, u0_lo)
tm.inject(D0pb_hi, D0pb_lo)
D0_lo.assign(D0pb_lo - b_lo)

np.save("day50-u-lo.npy", u0_lo.dat.data)
np.save("day50-D-lo.npy", D0_lo.dat.data)

### Do truth

u0_hi.dat.data[:] = np.load("truth-hi-u.npy")
D0_hi.dat.data[:] = np.load("truth-hi-D.npy")

D0pb_hi.assign(D0_hi + b_hi)
tm.inject(u0_hi, u0_lo)
tm.inject(D0pb_hi, D0pb_lo)
D0_lo.assign(D0pb_lo - b_lo)

np.save("truth-lo-u.npy", u0_lo.dat.data)
np.save("truth-lo-D.npy", D0_lo.dat.data)

# File("temphi.pvd").write(u0_hi, D0_hi)
# File("templo.pvd").write(u0_lo, D0_lo)
