from importlib.metadata import version
import types


def _print_Assignment_original(self, expr):
    """Original method from sympy 1.10.

    8fe2c879 introduced additiaonl calls to self._arrayify which
    are removed here."""
    return "%s = %s" % (
        self._print(expr.lhs),
        self._print(expr.rhs),
    )


def patch_assignment_printing(printer=None):
    """Fix assingment printing in sympy > 1.10.

    See
        https://github.com/sympy/sympy/issues/24236
    and
        https://github.com/sympy/sympy/pull/22950

    #22950 leads to assignments printed w/ einsum calls.
    As of sympy 1.12 this still is not fixed."""

    ver = version("sympy")
    minor = ver.split(".")[1]
    minor = int(minor)
    # Do nothing for sympy < 1.11
    if minor < 11:
        return

    # Patch specific printer instance
    if printer is not None:
        printer._print_Assignment = types.MethodType(
            _print_Assignment_original, printer
        )
    # Patch whole module
    else:
        # ArrayPrinter is not present in 1.10
        from sympy.printing.pycode import ArrayPrinter

        ArrayPrinter._print_Assignment = _print_Assignment_original
