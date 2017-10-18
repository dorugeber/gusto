from gusto import *
from firedrake import SpatialCoordinate, as_vector
from math import pi
import sys

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 3000., 4: 1500., 5: 750., 6: 375.}
    tmax = 5*day

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)
    domain = SphericalDomain(radius=R, refinement_level=ref_level)

    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])

    state = State(domain,
                  horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    x = SpatialCoordinate(domain.mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    Omega = parameters.Omega
    g = parameters.g
    Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    ueqn = VectorInvariant(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # build time stepper
    stepper = Timestepper(state, advected_fields, linear_solver,
                          sw_forcing)

    stepper.run(t=0, tmax=tmax)
