from fenics import *

#define FunctionSpace for velocity and pressure

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)


