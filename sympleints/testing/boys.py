from typing import Callable

import numpy as np
import scipy as sp
from scipy.special import factorial, factorial2


def boys_quad(n, x, err=1e-13):
    """Boys function from quadrature."""

    def inner(t):
        return t ** (2 * n) * np.exp(-x * t**2)

    y, err = sp.integrate.quad(inner, 0.0, 1.0, epsabs=err, epsrel=err)
    return y


def get_table(nmax, order, dx, xmax):
    nsteps = (xmax / dx) + 1
    # assert that mantissa of nsteps is 0.0
    assert (
        float(int(nsteps)) == nsteps
    ), f"{xmax=} must be an integer multiple of {dx=:.8e} but it isn't!"
    nsteps = int(nsteps)
    xs = np.linspace(0.0, xmax, num=nsteps)
    table = np.empty_like(xs)
    for i, x in enumerate(xs):
        table[i] = boys_quad(nmax + order, x)
    return xs, table


def boys_x0(n):
    """Boys-function when x equals 0.0."""
    return 1 / (2 * n + 1)


def boys_xlarge(n, x):
    """Boys-function for large x values (x >= 30 is sensible)."""
    _2n = 2 * n
    f2 = factorial2(_2n - 1) if n > 0 else 1.0
    return f2 / 2 ** (n + 1) * np.sqrt(np.pi / x ** (_2n + 1))


def get_boys_func(
    nmax: int, order: int = 6, dx: float = 0.1, xlarge: float = 30.0
) -> Callable[[int, float], float]:
    """Wrapper that returns Boys function object.

               1
               /
              |
              |          2
    F_n(x) =  |   2*n  -t x
              |  t    e      dt
              |
             /
             0

     Parameters
     ----------
     nmax
         Positive integer Maximum n value for which the Boys function can be evaluated.
     order
         Order of the Tayor-expansion that is used to evaluate the Boys-function, defaults
         to 6.
     dx
        Positive float; Distance between points on the grid.
     xlarge
        Cutoff value. For 0 < x < xlarge a Taylor expansion will be used to evaluate
        the boys function. For x >= xlarge the Boys-function is evaluated using the
        incomplete gamma function. For all n and x = 0.0 the integral is calculated
        analytically.

    Returns
    -------
    boys
        Callable F(n, x) that accepts two arguments: an integer 0 <= n <= nmax and a
        float 0.0 <= x. It returns the values of the Boys-function F_n at the given x.
    """
    # Pretabulated Boys-values on a grid for n = nmax + order
    xs, table = get_table(nmax, order, dx, xlarge)
    nxs = len(xs)

    # Build up full table for all n = {0, 1, ... nmax + order} via Downward recursion
    table_full = np.empty((nmax + order + 1, nxs))
    table_full[-1] = table
    exp_minus_xs = np.exp(-xs)
    _2xs = 2 * xs
    # Loop over all n-values
    for n in range(nmax + order - 1, -1, -1):
        table_full[n] = (_2xs * table_full[n + 1] + exp_minus_xs) / (2 * n + 1)
    # 1 / k!
    _1_factorials = 1 / factorial(np.arange(order, dtype=int))

    # Prefactors for Boys-values for large x > xlarge
    xlarge_prefact = np.empty(nmax + 1)
    for n in range(nmax + 1):
        xlarge_prefact[n] = (
            (factorial2(2 * n - 1) if n > 0 else 1.0) / 2 ** (n + 1) * np.sqrt(np.pi)
        )

    def boys_xlarge_precalc(n, x):
        return xlarge_prefact[n] * x ** (-n - 0.5)

    def boys_taylor(n, x):
        # Rounding to the nearest integer always yields a grid point before (or on) x
        # so we interpolate and don't extrapolate
        factor = int(1 / dx)
        ind0 = int(x * factor)
        # Closest grid point (on/before) x
        xg = ind0 * dx

        # Step length for Taylor expansion; depends on closest grid point
        dx_taylor = x - xg
        result = 0.0
        # The loop is somehow faster than the vectorized expression
        for k in range(order):
            result += table_full[n + k, ind0] * (-dx_taylor) ** k * _1_factorials[k]
        return result

    def boys(n: int, xs: float) -> float:
        def func(n, x):
            if x == 0.0:
                # Take values directly from table; should correspond to 1 / (2n + 1)
                return table_full[n, 0]
            elif x < xlarge:
                return boys_taylor(n, x)
            else:
                return boys_xlarge_precalc(n, x)

        if isinstance(xs, np.ndarray):
            boys_list = list()

            with np.nditer(xs) as it:
                for x in it:
                    boys_list.append(func(n, x))
            boys_table = np.reshape(boys_list, xs.shape)
            return boys_table
        else:
            return func(n, xs)

    return boys


# Calculation should take ~ 20 ms
# python -m timeit -n 25 "from sympleints.boys import boys"
boys = get_boys_func(nmax=8)
