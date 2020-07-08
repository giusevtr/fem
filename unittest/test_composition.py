import sys
sys.path.append("../private-pgm/src")
import util
import pytest

# class TestComposition:

@pytest.mark.parametrize("epsilon,eps0", [(0.1,0.003), (0.1,0.005), (0.1,0.007), (1, 0.001)])
def test_get_rounds(epsilon, eps0):
    # epsilon = 1
    delta = 1e-8
    # eps0 = 0.1
    T1 = util.get_rounds(epsilon, eps0, delta)
    T2 = util.get_rounds_zCDP(epsilon, eps0, delta)
    assert T1 == T2
