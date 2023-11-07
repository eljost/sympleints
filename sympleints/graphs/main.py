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
from sympleints.l_iters import ll_iter, lllaux_iter, schwarz_iter


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=("nucattr", "int2c2e", "int3c2e", "schwarz"))
    parser.add_argument("--lmax", type=int, default=L_MAX)
    parser.add_argument("--lauxmax", type=int, default=L_AUX_MAX)
    parser.add_argument("--do-plot", action="store_true")
    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])
    key = args.key
    lmax = args.lmax
    lauxmax = args.lauxmax
    do_plot = args.do_plot

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
        gen_integrals.append(generate_integral(L_tots, integral, do_plot=do_plot))
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
