from gusto import *
from firedrake import as_vector, SpatialCoordinate,\
    PeriodicRectangleMesh, ExtrudedMesh, \
    exp, cos, sin, cosh, sinh, tanh, pi
import sys

dt = 30.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 30*24*60*60.

##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh
columns = 30  # number of columns
L = 1000000.
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e4, quadrilateral=True)

# build 2D mesh by extruding the base mesh
nlayers = 30  # horizontal layers
H = 10000.  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

##############################################################################
# set up all the other things that state requires
##############################################################################
# Coriolis expression
f = 1.e-04
Omega = as_vector([0.,0.,f*0.5])
Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
dbdy = -1.0e-07

# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 3D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'rho', 'theta']

# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
output = OutputParameters(dirname='compressible_eady', dumpfreq=240,
                          dumplist=['u','rho','theta'],
                          perturbation_fields=['rho', 'theta'])

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = CompressibleEadyParameters(H=H, Nsq=Nsq, dbdy=dbdy, f=f)

# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), ExnerPi(), VerticalVelocity(),
                     KineticEnergy(), KineticEnergyV()]

# setup state, passing in the mesh, information on the required finite element
# function spaces and the classes above
state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              Coriolis=Omega,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

##############################################################################
# Initial conditions
##############################################################################
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# first setup the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
x, y, z = SpatialCoordinate(mesh)
g = parameters.g

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetaref = Tsurf*exp(Nsq*(z-H/2)/g)
theta_b = Function(Vt).interpolate(thetaref)
rho_b = Function(Vr)


# setup constants
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -7.5
Bu = 0.5
theta_exp = 30.*a*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)-n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
theta_pert = Function(Vt).interpolate(theta_exp)

# set theta0
theta0.interpolate(theta_b + theta_pert)

# hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)
compressible_hydrostatic_balance(state, theta0, rho0)

# balanced u
pi0 = compressible_eady_initial_u(state, theta_b, rho_b, u0)

state.initialise({'u':u0, 'rho':rho0, 'theta':theta0})
state.set_reference_profiles({'rho':rho_b, 'theta':theta_b})

##############################################################################
# Set up advection schemes
##############################################################################
# we need a DG funciton space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt, supg_params={"dg_direction":"horizontal"})

advection_dict = {}
advection_dict["u"] = ThetaMethod(state, u0, ueqn)
advection_dict["rho"] = SSPRK3(state, rho0, rhoeqn)
advection_dict["theta"] = SSPRK3(state, theta0, thetaeqn)

##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
# Set up linear solver
linear_solver_params = {'pc_type': 'fieldsplit',
                        'pc_fieldsplit_type': 'schur',
                        'ksp_type': 'gmres',
                        'ksp_monitor_true_residual': False,
                        'ksp_max_it': 100,
                        'ksp_gmres_restart': 50,
                        'pc_fieldsplit_schur_fact_type': 'FULL',
                        'pc_fieldsplit_schur_precondition': 'selfp',
                        'fieldsplit_0_ksp_type': 'preonly',
                        'fieldsplit_0_pc_type': 'bjacobi',
                        'fieldsplit_0_sub_pc_type': 'ilu',
                        'fieldsplit_1_ksp_type': 'preonly',
                        'fieldsplit_1_pc_type': 'gamg',
                        'fieldsplit_1_pc_gamg_sym_graph': True,
                        'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                        'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                        'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                        'fieldsplit_1_mg_levels_ksp_max_it': 5,
                        'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                        'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}


linear_solver = CompressibleSolver(state, params=linear_solver_params)

##############################################################################
# Set up forcing
##############################################################################
compressible_forcing = CompressibleEadyForcing(state, pi0=pi0,
                                               euler_poincare=False)

##############################################################################
# build time stepper
##############################################################################
stepper = Timestepper(state, advection_dict, linear_solver,
                      compressible_forcing, diagnostic_everydump=True)

##############################################################################
# Run!
##############################################################################
stepper.run(t=0, tmax=tmax)