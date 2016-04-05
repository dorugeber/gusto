from dcore import *
from firedrake import IcosahedralSphereMesh, Expression, SpatialCoordinate, \
    as_vector
from math import pi


def setup():

    refinements = 3  # number of horizontal cells = 20*(4^refinements)
    R = 1.
    dt = pi/3*0.001

    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinements)
    global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])",
                                "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
    mesh.init_cell_orientations(global_normal)

    fieldlist = ['u','D']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dumpfreq=150, dirname='tests/DGAdv')
    parameters = ShallowWaterParameters()

    state = ShallowWaterState(mesh, vertical_degree=None, horizontal_degree=2,
                              family="BDM",
                              timestepping=timestepping,
                              output=output,
                              parameters=parameters,
                              fieldlist=fieldlist)

    # interpolate initial conditions
    u0, Ddg = Function(state.V[0]), Function(state.V[1])
    x = SpatialCoordinate(mesh)
    uexpr = as_vector([-x[1], x[0], 0.0])
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[1],2))")

    u0.project(uexpr)
    Ddg.interpolate(Dexpr)

    state.initialise([u0, Ddg])

    return state, Ddg.function_space()


def run():

    state, Vdg = setup()

    dt = state.timestepping.dt
    tmax = pi/2.
    t = 0.
    Ddg_advection = DGAdvection(state, Vdg)

    state.xn.assign(state.x_init)
    xn_field = state.xn.split()[1]
    xnp1_field = state.xnp1.split()[1]
    Ddg_advection.ubar.assign(state.xn.split()[0])
    state.dump()

    while t < tmax - 0.5*dt:
        t += dt
        Ddg_advection.apply(xn_field, xnp1_field)
        state.xn.assign(state.xnp1)
        xn_field.assign(xnp1_field)
        state.dump()

    return state.xn.split()[1]


def test_dgadvection():

    D = run()
    Dend = Function(D.function_space())
    Dexpr = Expression("exp(-pow(x[2],2) - pow(x[0],2))")
    Dend.interpolate(Dexpr)
    Derr = Function(D.function_space()).assign(Dend - D)
    assert(Derr.dat.data.max() < 1.e-2)
    assert(abs(Derr.dat.data.max()) < 1.5e-2)
