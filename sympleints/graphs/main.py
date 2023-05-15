#!/usr/bin/env python3

import argparse
import itertools as it
import shutil
import sys
import time

from sympleints.config import L_AUX_MAX, L_MAX
from sympleints.graphs.defs.int_nucattr import get_nucattr
from sympleints.graphs.defs.int_2c2e import get_int_2c2e
from sympleints.graphs.defs.int_3c2e import get_int_3c2e
from sympleints.graphs.defs.int_4c2e import get_int_4c2e
from sympleints.graphs.generate import generate_integral
from sympleints.graphs.render import render_fortran_module


# Note the order (Lb, La) in the iterators below, ensuring Lb >= La.
# As the OS recursion relations build up angular momentum from left to right
# it is most economical to increase angular momentum at the left-most index first.
# Basically the wrappers below are thin wrappers around itertools-functions to
# yield column-major indices.


def ll_iter(lmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        yield (La, Lb)


def lllaux_iter(lmax, lauxmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        for Lc in range(lauxmax + 1):
            yield (La, Lb, Lc)


def schwarz_iter(lmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    # Yields total angular momenta for Schwarz integrals
    # (00|00), (10|10), (11|11) etc.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        yield La, Lb, La, Lb


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, default=L_MAX)
    parser.add_argument("--lauxmax", type=int, default=L_AUX_MAX)
    parser.add_argument("key", choices=("nucattr", "int2c2e", "int3c2e", "schwarz"))
    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])
    key = args.key
    lmax = args.lmax
    lauxmax = args.lauxmax

    iter_funcs = {
        # Nuclear attraction integrals
        "nucattr": (
            ll_iter(lmax),
            get_nucattr,
        ),
        # 2-center-2-electron integrals
        "int2c2e": (
            ll_iter(lauxmax),
            get_int_2c2e,
        ),
        # 3-center-2-electron integrals
        "int3c2e": (
            lllaux_iter(lmax, lauxmax),
            get_int_3c2e,
        ),
        # 4-center-2-electron integrals for Schwarz screening
        "schwarz": (
            schwarz_iter(lmax),
            get_int_4c2e,
        ),
    }
    L_tots_iter, int_func = iter_funcs[key]
    gen_integrals = list()
    start = time.time()
    for i, L_tots in enumerate(L_tots_iter):
        print(f"Integral {i:03d}")
        integral = int_func(L_tots)
        gen_integrals.append(generate_integral(L_tots, integral))
    gen_dur = time.time() - start

    start = time.time()
    last_key = "".join(map(str, L_tots))
    rendered = render_fortran_module(gen_integrals, lmax=lmax, lauxmax=lauxmax)
    render_dur = time.time() - start

    mod_fn = f"{integral.name}.f90"
    with open(mod_fn, "w") as handle:
        handle.write(rendered)
    shutil.copy(mod_fn, f"{integral.name}_{last_key}.f90")

    dur = gen_dur + render_dur
    print(f"Generation took {gen_dur:.1f} s")
    print(f" Rendering took {render_dur:.1f} s")
    print(f"Total {dur:.1f} s")


if __name__ == "__main__":
    run()
