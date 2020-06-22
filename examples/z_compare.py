from firedrake import *
import numpy as np

baseref = 2
# ref_hi = 5
ref_lo = 4

R = 6371220.0

assert ref_lo >= baseref

mesh0 = CubedSphereMesh(radius=R, refinement_level=baseref)
hierarchy = MeshHierarchy(mesh0, ref_lo-baseref)
mesh = hierarchy[ref_lo - baseref]

mesh.coordinates.dat.data[:] *= (R / np.linalg.norm(mesh.coordinates.dat.data, axis=1)).reshape(-1, 1)

x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

cell = mesh.ufl_cell().cellname()
V_elt = FiniteElement("RTCF", cell, 2, variant="equispaced")
W_elt = FiniteElement("DG", cell, 1, variant="equispaced")

V = FunctionSpace(mesh, V_elt)
W = FunctionSpace(mesh, W_elt)

truth_u = Function(V)
truth_D = Function(W)

model_u = Function(V)
model_D = Function(W)

truth_u.dat.data[:] = np.load("truth-lo-u.npy")
truth_D.dat.data[:] = np.load("truth-lo-D.npy")

model_u.dat.data[:] = np.load("double-u.npy")
model_D.dat.data[:] = np.load("double-D.npy")

err_u = Function(V).assign(truth_u - model_u)
err_D = Function(W).assign(truth_D - model_D)

l2err_u = sqrt(assemble(inner(err_u, err_u)*dx))/sqrt(assemble(inner(truth_u, truth_u)*dx))
l2err_D = sqrt(assemble(inner(err_D, err_D)*dx))/sqrt(assemble(inner(truth_D, truth_D)*dx))

print("L^2 u error:", l2err_u)
print("L^2 D error:", l2err_D)

File("temperr_u.pvd").write(truth_u, model_u, err_u)
File("temperr_D.pvd").write(truth_D, model_D, err_D)
