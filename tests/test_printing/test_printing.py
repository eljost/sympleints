import pytest
import sympy as sym
from sympy.codegen import Assignment
from sympy.printing.numpy import NumPyPrinter

from sympleints.patch_sympy import patch_assignment_printing


@pytest.fixture
def expr():
    x0, x1, a, b = sym.symbols("x:2 a b", real=True)
    lhs = x1
    rhs = x0 * a * b
    return Assignment(lhs, rhs)


def test_unpatched(expr):
    """No patch."""
    print_func = NumPyPrinter().doprint

    printed = print_func(expr)
    # print("unpatched", printed)
    #                  <= 1.10         1.0 < version <= 1.12
    assert printed in ("x1 = a*b*x0", 'x1 = numpy.einsum(",,", a, b, x0)')


def test_patch_printer(expr):
    """Patch only one printer object."""
    printer = NumPyPrinter()
    patch_assignment_printing(printer)
    print_func = printer.doprint

    printed = print_func(expr)
    # print("patched", printed)
    assert printed == "x1 = a*b*x0"


def test_patch_module(expr):
    """Patch whole module."""
    patch_assignment_printing()
    print_func = NumPyPrinter().doprint

    printed = print_func(expr)
    assert printed == "x1 = a*b*x0"
