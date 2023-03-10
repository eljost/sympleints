#!/usr/bin/env python3

import argparse
import itertools as it
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from jinja2 import Environment, PackageLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotext as ploplt
except ModuleNotFoundError:
    pass


from sympleints import bench_logger as logger


pd.set_option('display.float_format', '{:12.6e}'.format)


ENV = Environment(
    loader=PackageLoader("sympleints"),
    trim_blocks=True,
    lstrip_blocks=True,
)
TPL = ENV.get_template("fortran_bench.tpl")

KEYS = {
    "ovlp": ("ovlp3d", 1),
    "kin": ("kinetic3d", 1),
    "dpm": ("dipole3d", 3),
    "qpm": ("quadrupole3d", 6),
    # "coul": ("coulomb3d", 1),
    # "2c2e": ("int2c2e", 1),
    # "3c2e_sph": ("int3c2e_sph", -1),
}
KEYS_STR = ", ".join(KEYS.keys())


def compile(cwd, srcs, flags=None):
    if flags is None:
        flags = list()
    flags = " ".join(flags)
    args = f"gfortran {flags} {' '.join(srcs)}"
    _ = subprocess.check_output(args, cwd=cwd, shell=True, text=True)
    assert _ == ""


def run_benchmark(
    key,
    ncomponents,
    lmax,
    lauxmax,
    is_spherical,
    nprims=5,
    niters=10_000,
    flags=None,
    nmacro_iters=5,
):
    is_spherical_f = ".true." if is_spherical else ".false."
    src = TPL.render(
        lmax=lmax,
        lauxmax=-1,
        nprims=nprims,
        niters=niters,
        is_spherical=is_spherical_f,
        ncomponents=ncomponents,
        key=key,
    )
    id_ = f"{key} with flags: {flags}"

    macro_data = list()
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        f_fn = f"bench_{key}.f90"
        with open(tmp_path / f_fn, "w") as handle:
            handle.write(src)
        int_fn = f"{key}.f90"
        shutil.copy(int_fn, tmp_path)
        srcs = [int_fn, f_fn]
        logger.info(f"starting compilation of '{id_}'")
        compile(tmp_path, srcs, flags=flags)
        logger.info("finished compilation")

        for i in range(nmacro_iters):
            logger.info(f"starting run {i}/{nmacro_iters} of '{id_}' ... ")
            data = subprocess.check_output(
                "./a.out", cwd=tmp_path, shell=True, text=True
            )
            arr = np.array(data.strip().split(), dtype=float).reshape(
                -1, 4
            )  # Wont work for 3center!
            sarr = np.array(
                list(zip(*arr.T)),
                dtype=[("La", "i1"), ("Lb", "i1"), ("tot", "f8"), ("iter", "f8")],
            )
            macro_data.append(sarr)
            sys.stdout.flush()
    # Outside tmp_dir context mananger
    df = pd.DataFrame(np.concatenate(macro_data))
    return df


def mpl_plot_all_data(key, all_data):
    fig, ax = plt.subplots()
    label = None
    for flags, data in all_data.items():
        *_, tot_times, iter_times = data.T
        if label is None:
            label = list(zip(*np.array(_, dtype=int)))
            xs = np.arange(len(label))
        ax.plot(xs, tot_times, "o", label=f"tot: '{flags}'")
    ax.set_xticks(xs)
    ax.set_xticklabels(label, rotation="vertical")
    ax.set_xlabel("L")
    ax.set_ylabel("tot time / s")
    ax.legend()
    ax.set_title(key)
    fig.tight_layout()
    # plt.show()
    return fig


def plo_plot_all_data(key, dfs):
    ploplt.clear_figure()
    ploplt.plot_size(0.85 * ploplt.tw(), ploplt.th() / 2)
    label = None
    nflags = len(dfs)
    iter_series = list()
    all_flags = list()
    for i, (flags, df) in enumerate(dfs.items()):
        grouped = df.groupby(["La", "Lb"])
        mean = grouped.mean()
        sem = grouped.sem()
        if label is None:
            label = list(mean["tot"].keys())
        xs = nflags * np.arange(len(label)) + (i % nflags)
        lbl = f"tot: {flags}"
        # ploplt.error(xs, mean["tot"].array, yerr=sem["tot"].array, color=colors[i])
        # logger.info(f"{key}, {flags}, per iteration: {mean['iter']}")
        ploplt.scatter(xs, mean["iter"].array, label=lbl)
        iter_series.append(mean["iter"])
        all_flags.append(flags)
    iter_df = pd.concat(iter_series, axis=1, keys=all_flags)
    logger.info(f"{key}, {flags}:")
    logger.info(iter_df.to_string())
    iter_df.to_csv(f"{key}_iter_df.csv")
    xs = np.arange(nflags * mean.shape[0])
    rep_label = []
    for x in xs:
        if x % nflags == 0:
            lbl = label[x // nflags]
        else:
            lbl = ""
        rep_label.append(lbl)
    ploplt.xticks(xs, rep_label)
    ploplt.xlabel("Angular momenta")
    ploplt.ylabel("tot time / s")
    ploplt.title(key)
    fig = ploplt.build()
    print(fig)  # Only print to stdout; will be saved later
    return fig


def run_key(key, ncomponents, lmax, is_spherical, niters, nprims, custom_flags=None):
    if custom_flags is None:
        custom_flags = []

    flags = [
        ["-O2", "-O3", "-Ofast"],
        [
            "",
            "-march=skylake",
        ],
        # ["", "-ftree-vectorize"],
    ]

    flag_prods = list(it.product(*flags))
    all_flags = flag_prods + custom_flags

    dfs = dict()
    for flags in all_flags:
        flags_key = ", ".join(flags)

        df = run_benchmark(
            key,
            ncomponents=ncomponents,
            lmax=lmax,
            lauxmax=-1,
            is_spherical=is_spherical,
            niters=niters,
            nprims=nprims,
            flags=flags,
        )
        dfs[flags_key] = df
    return dfs


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--niters", type=int, default=250_000)
    parser.add_argument("--nprims", type=int, default=6)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--is_spherical", action="store_true")
    parser.add_argument(
        "--keys",
        nargs="+",
        help=f"Benchmark only certain integrals. Possible keys are: {KEYS_STR}. "
        "If not given, all possible integrals are benchmarked.",
    )

    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    lmax = args.lmax
    is_spherical = False
    niters = args.niters
    nprims = args.nprims
    is_spherical = args.is_spherical
    keys = args.keys
    cwd = Path(".")

    if keys is None:
        keys = KEYS.keys()

    # Loop over different integral types
    for key_ in keys:
        key, ncomponents = KEYS[key_]
        dfs = run_key(
            key,
            ncomponents=ncomponents,
            lmax=lmax,
            is_spherical=is_spherical,
            niters=niters,
            nprims=nprims,
        )
        # fig = mpl_plot_all_data(key, all_data)
        # fig.savefig(f"{key}.png")
        fig = plo_plot_all_data(key, dfs)
        ploplt.save_fig((cwd / f"{key}.html").absolute())


if __name__ == "__main__":
    run()
