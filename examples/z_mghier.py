from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, Min, FunctionSpace, MeshHierarchy,
                       Function, assemble, dx, FiniteElement, inject, prolong, File)
import sys
import numpy as np

day = 24.*60.*60.
hour = 60.*60.
ref_dt = {4: 450.0}
tmax = 5.0*day

assert len(sys.argv) > 1, "No bits given"
assert len(sys.argv) <= 4, "At most 3 refinement levels supported"
pbits = [int(foo) for foo in sys.argv[1:]]  # number of bits used on each level
# e.g., pbits = [9, 8, 7]
plevels = len(pbits)

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():
    dirname = "_".join([str(foo) for foo in pbits]) + "_sw_W5_ref%s_dt%s" % (ref_level, dt)
    mesh0 = CubedSphereMesh(radius=R, refinement_level=ref_level-2)
    hierarchy = MeshHierarchy(mesh0, 2)
    mesh = hierarchy[-1]
    mesh.coordinates.dat.data[:] *= (R / np.linalg.norm(mesh.coordinates.dat.data, axis=1)).reshape(-1, 1)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)

    output = OutputParameters(dirname=dirname,
                              dumplist_latlon=['D'],
                              dumpfreq=1,
                              log_level='INFO')

    diagnostic_fields = [Sum('D', 'topography')]

    state = State(mesh, horizontal_degree=1,
                  family="RTCF",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
                  fieldlist=fieldlist)

    cell = mesh.ufl_cell().cellname()
    DG_elt = FiniteElement("DG", cell, 1, variant="equispaced")

    Vdgs = [FunctionSpace(msh, DG_elt) for msh in hierarchy[-plevels:]]

    us_exact = [Function(fs, name="uexact_"+str(idx)) for idx, fs in enumerate(Vdgs)]
    us_prolonged = [Function(fs, name="uprolong_"+str(idx)) for idx, fs in enumerate(Vdgs)]
    us_decomp = [Function(fs, name="udecomp_"+str(idx)) for idx, fs in enumerate(Vdgs)]
    us_sum = [Function(fs, name="usum_"+str(idx)) for idx, fs in enumerate(Vdgs)]

    # calculate avg bits per DoF
    ndofs = []
    for ii in range(plevels):
        ndofs.append(len(us_decomp[ii].dat.data))
    tbits = 0
    for ii in range(plevels):
        tbits += pbits[ii]*ndofs[ii]
    avgbits = tbits/ndofs[-1]
    print("Bits:", pbits, avgbits)

    outfiles = [File("results/alt" + dirname + "/D"+str(idx)+".pvd") for idx, nbit in enumerate(pbits)]


    def roundfield(field, nbits):
        biggestnum = max(abs(np.max(field.dat.data[:])), abs(np.min(field.dat.data[:])))
        prec = 2.0**(np.ceil(np.log2(biggestnum)) - (nbits - 1))
        field.dat.data[:] = prec * np.round(field.dat.data[:]/prec)


    def roundhier(field, pbits):
        # copy into function list
        us_exact[-1].assign(field)

        # inject to coarsest level
        for ii in range(plevels-1, 0, -1):
            inject(us_exact[ii], us_exact[ii-1])

        # at coarsest level, round directly
        us_decomp[0].assign(us_exact[0])
        roundfield(us_decomp[0], pbits[0])
        us_sum[0].assign(us_decomp[0])

        # at each level...
        for ii in range(1, plevels):
            # prolong the previous sum
            prolong(us_sum[ii-1], us_prolonged[ii])
            # calculate the residual
            us_decomp[ii].assign(us_exact[ii] - us_prolonged[ii])
            # round this
            roundfield(us_decomp[ii], pbits[ii])
            # create new sum
            us_sum[ii].assign(us_prolonged[ii] + us_decomp[ii])
        
        # copy back into field data
        field.assign(us_sum[-1])

        # hacky - output here too
        for ii in range(plevels):
            outfiles[ii].write(us_decomp[ii])


    def roundstate(xn):
        xnu, xnD = xn.split()
        roundhier(xnD, pbits)
        

    # interpolate initial conditions
    u0 = state.fields('u')
    D0 = state.fields('D')
    x = SpatialCoordinate(mesh)
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
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
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    # Coriolis
    fexpr = 2*Omega*x[2]/R
    V = FunctionSpace(mesh, "CG", 1)
    f = state.fields("coriolis", V)
    f.interpolate(fexpr)  # Coriolis frequency (1/s)
    b = state.fields("topography", D0.function_space())
    b.interpolate(bexpr)

    u0.project(uexpr)
    D0.interpolate(Dexpr)
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
                            sw_forcing, round_fn=roundstate)

    stepper.run(t=0, tmax=tmax)

    D_double = Function(state.fields('D'))
    D_double.dat.data[:] = np.load("sw-double.npy")

    D_diff = Function(state.fields('D'))
    D_diff.assign(state.fields('D') - D_double)
    l2err = sqrt(assemble(D_diff*D_diff*dx))/sqrt(assemble(D_double*D_double*dx))
    print("Bits:", pbits, avgbits)
    print("Relative L^2 error:", l2err)
