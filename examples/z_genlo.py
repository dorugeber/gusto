from firedrake import (CubedSphereMesh, FunctionSpace, MeshHierarchy, inject,
                       restrict, prolong, SpatialCoordinate, Function, File)
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

V_hi = FunctionSpace(mesh_hi, "RTCF", 2)
W_hi = FunctionSpace(mesh_hi, "DQ", 1)

V_lo = FunctionSpace(mesh_lo, "RTCF", 2)
W_lo = FunctionSpace(mesh_lo, "DQ", 1)

u0_hi = Function(V_hi)
D0_hi = Function(W_hi)

u0_hi.dat.data[:] = np.load("day50-u.npy")
D0_hi.dat.data[:] = np.load("day50-D.npy")

u0_lo = Function(V_lo)
D0_lo = Function(W_lo)

# inject(u0_hi, u0_lo)
inject(D0_hi, D0_lo)

np.save("day50-u-lo.npy", u0_lo.dat.data)
np.save("day50-D-lo.npy", D0_lo.dat.data)

File("temphi.pvd").write(u0_hi, D0_hi)
File("templo.pvd").write(u0_lo, D0_lo)
