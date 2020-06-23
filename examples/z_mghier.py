from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate,
                       pi, sqrt, Min, FunctionSpace, MeshHierarchy,
                       Function, assemble, dx, FiniteElement, TransferManager, File,
                       inner)
import sys
import numpy as np

baseref = 2
day = 24.*60.*60.
hour = 60.*60.
ref_dt = {4: 450.0}
dfreq = 8
tmax = 10.0*day

assert len(sys.argv) > 2, "Give number of levels, followed by u and D bits on each level"
assert len(sys.argv) <= 8, "At most 3 refinement levels supported"

plevels = int(sys.argv[1])
assert plevels in (1, 2, 3), "Number of levels must be 1, 2, or 3"
assert len(sys.argv) == 2*plevels + 2, "Wrong number of arguments"

allbits = [int(foo) for foo in sys.argv[2:]]  # e.g. [8, 9, 7, 7]
ubits = allbits[::2]  # e.g., [8, 7]
Dbits = allbits[1::2]  # e.g., [9, 7]


# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

for ref_level, dt in ref_dt.items():
    assert ref_level >= baseref
    dirname = "comp_" + "_".join([str(foo) for foo in allbits])
    mesh0 = CubedSphereMesh(radius=R, refinement_level=baseref)
    hierarchy = MeshHierarchy(mesh0, ref_level-baseref)

    for msh in hierarchy[-plevels:]:
        msh.coordinates.dat.data[:] *= (R / np.linalg.norm(msh.coordinates.dat.data, axis=1)).reshape(-1, 1)
        x = SpatialCoordinate(msh)
        msh.init_cell_orientations(x)

    mesh = hierarchy[-1]
    timestepping = TimesteppingParameters(dt=dt)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=dfreq,
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
    V_elt = FiniteElement("RTCF", cell, 2, variant="equispaced")
    W_elt = FiniteElement("DG", cell, 1, variant="equispaced")

    Vs = [FunctionSpace(msh, V_elt) for msh in hierarchy[-plevels:]]
    Ws = [FunctionSpace(msh, W_elt) for msh in hierarchy[-plevels:]]

    us_exact = [Function(fs, name="uexact_"+str(idx)) for idx, fs in enumerate(Vs)]
    us_prolonged = [Function(fs, name="uprolong_"+str(idx)) for idx, fs in enumerate(Vs)]
    us_decomp = [Function(fs, name="udecomp_"+str(idx)) for idx, fs in enumerate(Vs)]
    us_sum = [Function(fs, name="usum_"+str(idx)) for idx, fs in enumerate(Vs)]

    Ds_exact = [Function(fs, name="Dexact_"+str(idx)) for idx, fs in enumerate(Ws)]
    Ds_prolonged = [Function(fs, name="Dprolong_"+str(idx)) for idx, fs in enumerate(Ws)]
    Ds_decomp = [Function(fs, name="Ddecomp_"+str(idx)) for idx, fs in enumerate(Ws)]
    Ds_sum = [Function(fs, name="Dsum_"+str(idx)) for idx, fs in enumerate(Ws)]

    # calculate avg bits per DoF
    udofs = []
    Ddofs = []
    for ii in range(plevels):
        udofs.append(len(us_decomp[ii].dat.data))
        Ddofs.append(len(Ds_decomp[ii].dat.data))
    tbits = 0
    for ii in range(plevels):
        tbits += ubits[ii]*udofs[ii]
        tbits += Dbits[ii]*Ddofs[ii]
    avgbits = tbits/(udofs[-1]+Ddofs[-1])
    print("Bits:", allbits, avgbits)

    dcount = 0
    outfiles = [File("results/alt" + dirname + "/uD"+str(idx)+".pvd") for idx, nbit in enumerate(ubits)]
    tm = TransferManager()

    def roundfield(field, nbits):
        biggestnum = max(abs(np.max(field.dat.data[:])), abs(np.min(field.dat.data[:])))
        prec = 2.0**(np.ceil(np.log2(biggestnum)) - (nbits - 1))
        field.dat.data[:] = prec * np.round(field.dat.data[:]/prec)

    def roundstate(xn):
        global dcount
        xnu, xnD = xn.split()

        # 1. Velocity
        # copy into function list
        us_exact[-1].assign(xnu)

        # inject to coarsest level
        for ii in range(plevels-1, 0, -1):
            tm.inject(us_exact[ii], us_exact[ii-1])

        # at coarsest level, round directly
        us_decomp[0].assign(us_exact[0])
        roundfield(us_decomp[0], ubits[0])
        us_sum[0].assign(us_decomp[0])

        # at each level...
        for ii in range(1, plevels):
            # prolong the previous sum
            tm.prolong(us_sum[ii-1], us_prolonged[ii])
            # calculate the residual
            us_decomp[ii].assign(us_exact[ii] - us_prolonged[ii])
            # round this
            roundfield(us_decomp[ii], ubits[ii])
            # create new sum
            us_sum[ii].assign(us_prolonged[ii] + us_decomp[ii])

        # copy back into field data
        xnu.assign(us_sum[-1])

        # 2. Depth
        # copy into function list
        Ds_exact[-1].assign(xnD)

        # inject to coarsest level
        for ii in range(plevels-1, 0, -1):
            tm.inject(Ds_exact[ii], Ds_exact[ii-1])

        # at coarsest level, round directly
        Ds_decomp[0].assign(Ds_exact[0])
        roundfield(Ds_decomp[0], Dbits[0])
        Ds_sum[0].assign(Ds_decomp[0])

        # at each level...
        for ii in range(1, plevels):
            # prolong the previous sum
            tm.prolong(Ds_sum[ii-1], Ds_prolonged[ii])
            # calculate the residual
            Ds_decomp[ii].assign(Ds_exact[ii] - Ds_prolonged[ii])
            # round this
            roundfield(Ds_decomp[ii], Dbits[ii])
            # create new sum
            Ds_sum[ii].assign(Ds_prolonged[ii] + Ds_decomp[ii])

        # copy back into field data
        xnD.assign(Ds_sum[-1])

        # hacky - output here too
        if dcount % dfreq == 0:
            for ii in range(plevels):
                outfiles[ii].write(us_decomp[ii], Ds_decomp[ii])
        dcount += 1

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

    u0.dat.data[:] = np.load("day50-u-lo.npy")
    D0.dat.data[:] = np.load("day50-D-lo.npy")

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

    # Analyse output...
    # Load in truth and double
    truth_u = Function(state.fields('u'))
    truth_D = Function(state.fields('D'))

    model_u = Function(state.fields('u'))
    model_D = Function(state.fields('D'))

    truth_u.dat.data[:] = np.load("truth-lo-u.npy")
    truth_D.dat.data[:] = np.load("truth-lo-D.npy")

    model_u.dat.data[:] = np.load("double-u.npy")
    model_D.dat.data[:] = np.load("double-D.npy")

    # Calculate errors
    err1_u = Function(state.fields('u')).assign(state.fields('u') - model_u)
    err1_D = Function(state.fields('D')).assign(state.fields('D') - model_D)

    err2_u = Function(state.fields('u')).assign(state.fields('u') - truth_u)
    err2_D = Function(state.fields('D')).assign(state.fields('D') - truth_D)

    l2err1_u = sqrt(assemble(inner(err1_u, err1_u)*dx))/sqrt(assemble(inner(model_u, model_u)*dx))
    l2err1_D = sqrt(assemble(inner(err1_D, err1_D)*dx))/sqrt(assemble(inner(model_D, model_D)*dx))

    l2err2_u = sqrt(assemble(inner(err2_u, err2_u)*dx))/sqrt(assemble(inner(truth_u, truth_u)*dx))
    l2err2_D = sqrt(assemble(inner(err2_D, err2_D)*dx))/sqrt(assemble(inner(truth_D, truth_D)*dx))

    print("L^2 errors vs double:", l2err1_u, l2err1_D)
    print("L^2 errors vs truth:", l2err2_u, l2err2_D)
    print("Bits:", allbits, avgbits)
