from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt,
                       TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver, DirichletBC)
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 2.
    deltax = 1000.
else:
    deltax = 250.
    tmax = 1500.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
diffusion = True

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='unsaturated_bubble', dumpfreq=20, dumplist=['u', 'theta'], perturbation_fields=['theta', 'water_v'], log_level='INFO')
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [Theta_e(), Temperature(), Dewpoint(), RelativeHumidity()]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("water_v", theta0.function_space())
water_c0 = state.fields("water_c", theta0.function_space())
moisture = ["water_v", "water_c"]

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
x, z = SpatialCoordinate(mesh)
quadrature_degree = (5, 5)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 300.0
Ttop = 350.0
humidity = 0.6
theta_d = Function(Vt).interpolate(Tsurf + (Ttop - Tsurf) * (z / H) ** 2.0)
RH = Function(Vt).assign(humidity)

# Calculate hydrostatic fields
unsaturated_hydrostatic_balance(state, theta_d, RH)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(conditional(r > rc,
                                                  0.0,
                                                  Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

# define initial theta
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dxp
L = gamma * (rho_b * theta_b / theta0) * dxp
rho_problem = LinearVariationalProblem(a, L, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

water_c0.assign(0.0)

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0),
                  ('water_v', water_v0),
                  ('water_c', water_c0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('water_v', water_vb)])

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

advected_fields = [('u', ThetaMethod(state, u0, ueqn)),
                   ('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn)),
                   ('water_v', SSPRK3(state, water_v0, thetaeqn)),
                   ('water_c', SSPRK3(state, water_c0, thetaeqn))]

linear_solver = HybridizedCompressibleSolver(state, moisture=moisture)

# Set up forcing
compressible_forcing = CompressibleForcing(state, moisture=moisture)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = [Condensation(state)]

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
