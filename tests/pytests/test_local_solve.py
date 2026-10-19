from ngsolve import *
from xfem import *
from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square
from ngsolve.meshes import MakeStructured2DMesh

import pytest

@pytest.mark.parametrize("space", [H1,L2])
@pytest.mark.parametrize("order", [1,2,3])
@pytest.mark.parametrize("aggregate", [False, True])

def test_patch_local_solve(space,order,aggregate):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = space(mesh, order=order)

    EA = ElementAggregation(mesh)
    roots = BitArray(mesh.ne)
    roots.Set()
    if aggregate:
        roots.Clear()
        roots[0] = True
    EA.Update(roots, ~roots)

    #print('non trivial patch elements\n', EA.els_in_nontrivial_patch)
    #print('elements in trivial patches\n', EA.els_in_trivial_patch)

    gfu = GridFunction(fes)
    gfu.Set(x**3 + sin(y))

    u,v = fes.TnT()
    bf = u*v*dx
    lf = gfu*v*dx

    gfupatch = GridFunction(fes)
    gfupatch.vec.data = PatchwiseSolve(EA,fes,bf,lf)

    Draw(gfu,mesh,"gfuset")
    Draw(gfupatch,mesh,"gfupatch")
    diff = sqrt(Integrate(InnerProduct(gfu-gfupatch,gfu-gfupatch) * dx ,mesh))
    assert(diff < 1e-12)


def test_extension_keeps_root_first():
    # The leaf has a lower element number than its root. Extension assembly
    # must still eliminate the leaf, keeping the root and singleton DoFs.
    mesh = MakeStructured2DMesh(nx=2, ny=1, quads=False)
    roots = BitArray([False, True, True, True])
    aggregation = ElementAggregation(mesh, roots, ~roots)
    fes = L2(mesh, order=0, dgjumps=True)
    u, v = fes.TnT()
    embedding = ExtensionEmbedding(
        aggregation, fes, (u-u.Other())*(v-v.Other())*dx(skeleton=True))
    assert (embedding.height, embedding.width) == (4, 3)
    coefficients = embedding.CreateRowVector()
    coefficients.FV().NumPy()[:] = [2, 3, 4]
    expected = embedding.CreateColVector()
    expected.FV().NumPy()[:] = [2, 2, 3, 4]
    assert Norm(embedding*coefficients-expected) < 1e-12

    # Updating to singleton patches must discard the old patch/facet tables.
    roots.Set()
    aggregation.Update(roots, ~roots)
    identity = ExtensionEmbedding(aggregation, fes, u*v*dx)
    coefficients = identity.CreateRowVector()
    coefficients.FV().NumPy()[:] = [1, 2, 3, 4]
    assert Norm(identity*coefficients-coefficients) < 1e-12


def test_root_without_assigned_leaf():
    mesh = MakeStructured2DMesh(nx=2, ny=1, quads=False)
    roots = BitArray([True, True, False, True])
    aggregation = ElementAggregation(mesh, roots, ~roots)
    fes = L2(mesh, order=0, dgjumps=True)
    u, v = fes.TnT()
    embedding = ExtensionEmbedding(
        aggregation, fes, (u-u.Other())*(v-v.Other())*dx(skeleton=True))
    assert (embedding.height, embedding.width) == (4, 3)
    coefficients = embedding.CreateRowVector()
    coefficients[:] = 1
    expected = embedding.CreateColVector()
    expected[:] = 1
    assert Norm(embedding*coefficients-expected) < 1e-12
