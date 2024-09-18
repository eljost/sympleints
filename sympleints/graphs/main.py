#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import time

from pathos.pools import ProcessPool

from sympleints.config import L_AUX_MAX, L_MAX
from sympleints.graphs.defs.int_nucattr import get_nucattr
from sympleints.graphs.defs.int_2c2e import get_int_2c2e
from sympleints.graphs.defs.int_3c2e import get_int_3c2e
from sympleints.graphs.defs.int_4c2e import get_int_4c2e
from sympleints.graphs.generate import generate_integral
from sympleints.graphs.render import (
    render_fortran_integral,
    render_fortran_module_from_rendered,
)
from sympleints.l_iters import ll_iter, lllaux_iter, schwarz_iter


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=("nucattr", "int2c2e", "int3c2e", "schwarz"))
    parser.add_argument("--lmax", type=int, default=L_MAX)
    parser.add_argument("--lauxmax", type=int, default=L_AUX_MAX)
    parser.add_argument("--do-plot", action="store_true")
    parser.add_argument(
        "--out-dir", default=".", help="Directory where the integral are written."
    )
    parser.add_argument(
        "--low-first",
        action="store_true",
        help="Start with low angular momenta. Useful for debugging. Starting with "
        "higher angular momenta is faster though.",
    )
    parser.add_argument("--pal", type=int, default=1)
    return parser.parse_args(args)


def gen_wrapper(L_tots, kwds):
    do_plot = kwds["do_plot"]
    int_func = kwds["int_func"]
    integral_ = int_func(L_tots)
    integral = generate_integral(L_tots, integral_, do_plot=do_plot)
    funcs_rendered, L_tots_rendered = render_fortran_integral(integral)
    return funcs_rendered, L_tots_rendered


def run():
    args = parse_args(sys.argv[1:])
    key = args.key
    lmax = args.lmax
    lauxmax = args.lauxmax
    do_plot = args.do_plot
    out_dir = Path(args.out_dir).absolute()
    pal = args.pal
    high_first = not args.low_first

    if not out_dir.exists():
        out_dir.mkdir()

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
    L_tots_iter = list(L_tots_iter)
    if high_first:
        L_tots_iter = L_tots_iter[::-1]

    kwds = {
        "int_func": int_func,
        "do_plot": do_plot,
    }
    gen_dur = time.time()
    # In parallel
    if pal > 1:
        kwds_repeated = [kwds] * len(L_tots_iter)
        pool = ProcessPool(nodes=pal)
        results = pool.map(gen_wrapper, L_tots_iter, kwds_repeated)
    # Serial
    else:
        results = [gen_wrapper(L_tots, kwds) for L_tots in L_tots_iter]
    gen_dur = time.time() - gen_dur
    if high_first:
        results = results[::-1]

    # Unpack rendered functions and associated L_tots
    funcs = list()
    L_tots = list()
    for res_func, res_L_tots in results:
        funcs.extend(res_func)
        L_tots.extend(res_L_tots)

    dummy_integral = int_func(L_tots_iter[0])
    name = dummy_integral.name
    render_dur = time.time()
    rendered = render_fortran_module_from_rendered(
        name, lmax=lmax, lauxmax=lauxmax, funcs=funcs, L_tots=L_tots
    )
    render_dur = time.time() - render_dur

    mod_fn = f"{name}.f90"

    with open(out_dir / mod_fn, "w") as handle:
        handle.write(rendered)

    dur = gen_dur + render_dur
    print(f"Generation took {gen_dur:.1f} s")
    print(f" Rendering took {render_dur:.1f} s")
    print(f"Total {dur:.1f} s")


if __name__ == "__main__":
    run()
