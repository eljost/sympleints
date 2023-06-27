from enum import Enum
import functools

import sympy as sym


"""
Regular solid harmonics, based on formulas from wikipedia
    https://en.wikipedia.org/wiki/Solid_harmonics#Real_form
"""

# Imaginary unit
Im = sym.I
# Factorial function
fact = sym.factorial
# Binomical coefficient
binom = sym.binomial
HALF = sym.Rational(1, 2)
k = sym.symbols("k", integer=True)
x, y, z, r = sym.symbols("x y z r", real=True)
r2sub = x**2 + y**2 + z**2
# Stone: 0, +1, -1, +2, -2, etc.
# Natural: -l, -l + 1, ... 0, +1, ... +l
mOrder = Enum("mOrder", ["STONE", "NATURAL"])
DEFAULT_ORDER = mOrder.NATURAL


def Am(m, x, y):
    return HALF * ((x + Im * y) ** m + (x - Im * y) ** m)


def Bm(m, x, y):
    return 1 / (2 * Im) * ((x + Im * y) ** m - (x - Im * y) ** m)


def gamma(l, m, k):
    return (
        (-1) ** k
        * 2 ** (-l)
        * binom(l, k)
        * binom(2 * l - 2 * k, l)
        * fact(l - 2 * k)
        / fact(l - 2 * k - m)
    )


def zpart(l, m, z):
    limit = sym.floor((l - m) / 2)
    return sym.Sum(
        gamma(l, m, k) * r ** (2 * k) * z ** (l - 2 * k - m), (k, 0, limit)
    ).doit()


def C(l, m, x, y, z):
    krond = int(m == 0)
    prefact = ((2 - krond) * fact(l - m) / fact(l + m)) ** HALF
    result = prefact * zpart(l, m, z) * Am(m, x, y)
    return result.simplify()


def S(l, m, x, y, z):
    prefact = (2 * fact(l - m) / fact(l + m)) ** HALF
    result = prefact * zpart(l, m, z) * Bm(m, x, y)
    return result.simplify()


def Rlm(l, m, x, y, z):
    if m < 0:
        func = S
        m = abs(m)
    else:
        func = C
    return func(l, m, x, y, z)


@functools.cache
def Rlm_poly(l, m):
    rlm = Rlm(l, m, x, y, z)
    # Expand r² terms into x² + y² + z²
    rlm = rlm.subs(r**2, r2sub)
    rlm = rlm.expand().simplify()
    rlm_poly = sym.Poly(rlm, x, y, z, domain=sym.RR)
    return rlm_poly


def get_m_iter(l, order: mOrder = DEFAULT_ORDER):
    """Yields m in order 0, +1, -1, +2, -2, ...

    Ordering as in 'The Theory of Intermolecular Forces' by Anthony Stone (AS).

    TODO: take argument that determines the order of m, e.g. like used by AS
    or from -l to +l.
    """
    if order == mOrder.STONE:
        for m in range(l + 1):
            yield m
            if m > 0:
                yield -m
    elif order == mOrder.NATURAL:
        for m in range(-l, l + 1):
            yield m
    else:
        raise Exception(f"Unknown ordering '{order}'! Valid orderings are {mOrder}")


def Rlm_polys_up_to_l_iter(lmax, order: mOrder = DEFAULT_ORDER):
    for l in range(lmax + 1):
        m_iter = get_m_iter(l, order)  # 0, +1, -1, +2, -2, ...
        for m in m_iter:
            poly = Rlm_poly(l, m)
            yield poly, (l, m)


# def Rlm_polys_up_to_l(lmax, order=mOrder.STONE):
# return list(Rlm_polys_up_to_l(lmax, order))


def get_stone_name(l, m):
    m_str = f"{abs(m)}" + ("c" if m > 0 else "s")
    lm_str = f"R_{l}{m_str}"
    return lm_str


def test_poly_iter(lmax=2):
    for poly, (l, m) in Rlm_polys_up_to_l_iter(lmax):
        name = get_stone_name(l, m)
        print(name, poly.terms())


if __name__ == "__main__":
    test_poly_iter(2)
