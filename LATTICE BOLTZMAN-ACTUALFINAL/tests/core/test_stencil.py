import numpy as np
import pytest
from pytest import approx
from lbm.core.stencil import Stencil

# list of stencil, every element is the tuple (d, q)
list_of_stencils = [(2, 9), (3, 15), (3, 19), (3, 27)]


def test_init_stencil_fail():
    with pytest.raises(Exception):
        Stencil(d=2, q=99)
    with pytest.raises(Exception):
        Stencil(d=4, q=9)


@pytest.mark.parametrize("d,q", list_of_stencils)
def test_init_stencil(d, q):
    stencil = Stencil(d=d, q=q)
    assert d == stencil.d
    assert q == stencil.q
    assert q == len(stencil.w)
    assert q == len(stencil.c)


@pytest.mark.parametrize("d,q", list_of_stencils)
def test_sum_of_weights(d, q):
    stencil = Stencil(d=d, q=q)
    assert 1.0 == approx(np.sum(stencil.w))


@pytest.mark.parametrize("d,q", list_of_stencils)
def test_opposite_population_index(d, q):
    stencil = Stencil(d=d, q=q)
    for i in range(stencil.q):
        c_sum = stencil.c[i] + stencil.c[stencil.i_opp[i]]
        # The sum of a velocity vector and its opposite should be zero
        for j in range(stencil.d):
            assert c_sum[j] == np.zeros_like(c_sum[j])


@pytest.mark.parametrize("d,q", list_of_stencils)
def test_find_population_index(d, q):
    stencil = Stencil(d=d, q=q)
    idx = 0
    for ci in stencil.c:
        assert idx == stencil.c_pop(ci)
        idx += 1


def test_symmetry_normal_d2q9():
    stencil = Stencil(2, 9)
    normal = [0, 1]
    n_idx = stencil.c_pop(normal)  # normal index
    assert [0, 0] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([0, 0])]])
    assert [1, 0] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([1, 0])]])
    assert [0, -1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([0, 1])]])
    assert [-1, 0] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([-1, 0])]])
    assert [0, 1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([0, -1])]])
    assert [1, -1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([1, 1])]])
    assert [-1, -1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([-1, 1])]])
    assert [-1, 1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([-1, -1])]])
    assert [1, 1] == approx(stencil.c[stencil.i_sym[n_idx, stencil.c_pop([1, -1])]])
