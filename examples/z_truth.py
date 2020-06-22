from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate, pi, sqrt,
                       Min, FunctionSpace, MeshHierarchy)
import numpy as np

baseref = 2
day = 24.*60.*60.
hour = 60.*60.
ref_dt = {5: 225.0}
tmax = 10.0*day

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():
    assert ref_level >= baseref
    dirname = "truth_ref%s" % ref_level
    mesh0 = CubedSphereMesh(radius=R, refinement_level=baseref)
    hierarchy = MeshHierarchy(mesh0, ref_level-baseref)
    mesh = hierarchy[-1]
    mesh.coordinates.dat.data[:] *= (R / np.linalg.norm(mesh.coordinates.dat.data, axis=1)).reshape(-1, 1)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=16,
                              log_level='INFO')

    diagnostic_fields = [Sum('D', 'topography')]

    state = State(mesh, horizontal_degree=1,
                  family="RTCF",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    # set up b and load in initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)
    theta, lamda = latlon_coords(mesh)
    Omega = parameters.Omega
    g = parameters.g
    Rsq = R**2
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = Min(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)

    # Coriolis
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)
    b = state.fields("topography", D0.function_space())
    b.interpolate(bexpr)

    u0.dat.data[:] = np.load("day50-u.npy")
    D0.dat.data[:] = np.load("day50-D.npy")

    state.initialise([('u', u0),
                      ('D', D0)])

    ueqn = AdvectionEquation(state, u0.function_space(), vector_manifold=True)
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            sw_forcing)

    stepper.run(t=0, tmax=tmax)

    # output after 10 days
    np.save("truth-hi-u.npy", state.fields('u').dat.data)
    np.save("truth-hi-D.npy", state.fields('D').dat.data)
