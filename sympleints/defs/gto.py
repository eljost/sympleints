import functools

from sympy import exp, Function

from sympleints import shell_iter


class CartGTO3d(Function):
    """3D Cartesian Gaussian function; not normalized."""

    @classmethod
    @functools.cache
    def eval(cls, i, j, k, a, Xa, Ya, Za):
        Xa2 = Xa**2
        Ya2 = Ya**2
        Za2 = Za**2
        return (Xa**i) * (Ya**j) * (Za**k) * exp(-a * (Xa2 + Ya2 + Za2))


class CartGTOShell(Function):
    @classmethod
    def eval(cls, La_tot, a, Xa, Ya, Za):
        exprs = [CartGTO3d(*La, a, Xa, Ya, Za) for La, in shell_iter((La_tot,))]
        # print(CartGTO3d.eval.cache_info())
        return exprs
