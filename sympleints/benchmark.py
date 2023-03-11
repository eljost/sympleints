#!/usr/bin/env python3

import argparse
import itertools as it
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from jinja2 import Environment, PackageLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotext as ploplt
except ModuleNotFoundError:
    pass

from sympleints import bench_logger as logger


sns.set_theme()
pd.set_option("display.float_format", "{:12.6e}".format)


ENV = Environment(
    loader=PackageLoader("sympleints"),
    trim_blocks=True,
    lstrip_blocks=True,
)
TPL = ENV.get_template("fortran_bench.tpl")

_KEYS = {
    "ovlp": ("ovlp3d", 1),
    "kin": ("kinetic3d", 1),
    "dpm": ("dipole3d", 3),
    "qpm": ("quadrupole3d", 6),
}
BOYS_KEYS = {
    "coul": ("coulomb3d", 1),
    "2c2e": ("int2c2e3d", 1),
    "3c2e_sph": ("int3c2e3d_sph", -1),
}
KEYS = _KEYS | BOYS_KEYS
NEED_BOYS = list([k for _, (k, _) in BOYS_KEYS.items()])
KEYS_STR = ", ".join(KEYS.keys())
BOYS_PATH = Path("/home/johannes/tmp/409_meson3/fympleints/ints")


def compile(cwd, srcs, flags=None):
    if flags is None:
        flags = list()
    flags = " ".join(flags)
    srcs = map(str, srcs)
    args = f"gfortran {flags} {' '.join(srcs)}"
    _ = subprocess.check_output(args, cwd=cwd, shell=True, text=True)
    assert _ == ""  # Check return code
    return args


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
    need_boys = key in NEED_BOYS
    if ncomponents < 0:
        ndim = 3
    else:
        ndim = 2
    angmoms = ["La", "Lb", "Lc"][:ndim]
    angmom_fields = [(am, "i1") for am in angmoms]
    src = TPL.render(
        lmax=lmax,
        lauxmax=lauxmax,
        nprims=nprims,
        niters=niters,
        is_spherical=is_spherical_f,
        ncomponents=ncomponents,
        ndim=ndim,
        key=key,
        need_boys=need_boys,
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
        if need_boys:
            srcs = [
                BOYS_PATH / "boys_data.f90",
                BOYS_PATH / "boys.f90",
            ] + srcs

        logger.info(f"starting compilation of '{id_}'")
        compile_cmd = compile(tmp_path, srcs, flags=flags)
        logger.info("finished compilation")

        # Dump benchmark file with compile cmd in first line 
        with open(f_fn, "w") as handle:
            handle.write("\n\n".join((f"! {compile_cmd}", src)))

        for i in range(nmacro_iters):
            logger.info(f"starting run {i}/{nmacro_iters} of '{id_}' ... ")
            start = time.time()
            data = subprocess.check_output(
                "./a.out", cwd=tmp_path, shell=True, text=True
            )
            dur = time.time() - start
            logger.info(f"\t ... the run took {dur:12.6e} s.")
            arr = np.array(data.strip().split(), dtype=float).reshape(
                -1, ndim + 2
            )  # Wont work for 3center!
            # angmom_fields = [("La", "i1"), ("Lb", "i1"), ("Lc", "i1")][:ndim]
            sarr = np.array(
                list(zip(*arr.T)),
                dtype=[*angmom_fields, ("tot", "f8"), ("iter", "f8")],
            )
            macro_data.append(sarr)
            sys.stdout.flush()

    # Outside tmp_dir context mananger
    df = pd.DataFrame(np.concatenate(macro_data))
    df.set_index(angmoms, inplace=True)
    return df


def pandas_plot(key, dfs, ndim):
    df = pd.concat(dfs, axis=1)
    df.to_csv(f"{key}_df.csv")
    gby = ["La", "Lb"] + ([] if ndim == 2 else ["Lc"])
    grouped = df.groupby(gby)
    mean = grouped.mean()
    std = grouped.std()
    iter_slice = pd.IndexSlice[:, "iter"]
    iter_means = mean.loc[:, iter_slice]
    iter_std = std.loc[:, iter_slice]
    fig, ax = plt.subplots(figsize=(16, 8))
    iter_means.plot.bar(yerr=iter_std, ax=ax)
    min_means = iter_means.min(axis=1)
    max_means = iter_means.max(axis=1)
    for i, (_, iter_time) in enumerate(min_means.items()):
        ax.annotate(f"{iter_time:.4e}", xy=(i, 1.05 * max_means.values[i]), ha="center")
    ax.set_xlabel("Angular momenta")
    ax.set_ylabel("t per iteration / s")
    ax.set_title(key)
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
        # sem = grouped.sem()
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


def run_key(
    key, ncomponents, lmax, lauxmax, is_spherical, niters, nprims, flag_lists,

):
    dfs = dict()
    for flags in it.product(*flag_lists):
        flags_key = ", ".join(flags)

        df = run_benchmark(
            key,
            ncomponents=ncomponents,
            lmax=lmax,
            lauxmax=lauxmax,
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
    parser.add_argument("--lauxmax", type=int, default=2)
    parser.add_argument("--is_spherical", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument(
        "--keys",
        nargs="+",
        help=f"Benchmark only certain integrals. Possible keys are: {KEYS_STR}. "
        "If not given, all possible integrals are benchmarked.",
    )
    parser.add_argument("--quick", action="store_true")

    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    lmax = args.lmax
    lauxmax = args.lauxmax
    is_spherical = False
    niters = args.niters
    nprims = args.nprims
    is_spherical = args.is_spherical
    keys = args.keys
    skip3d = args.skip_3d
    quick = args.quick

    cwd = Path(".")

    if keys is None:
        keys = KEYS.keys()
    keys = list(keys)
    if skip3d:
        try:
            keys.remove("3c2e_sph")
        except ValueError:
            print("No 3d integrals requested!")

    flag_lists = [
        ["-O2", "-O3", "-Ofast"],
        [
            "",
            "-march=skylake",
        ],
    ]
    if quick:
        flag_lists = [["-O0"]]

    # Loop over different integral types
    for key_ in keys:
        key, ncomponents = KEYS[key_]
        ndim = 2 if ncomponents > 0 else 3
        dfs = run_key(
            key,
            ncomponents=ncomponents,
            lmax=lmax,
            lauxmax=lauxmax,
            is_spherical=is_spherical,
            niters=niters,
            nprims=nprims,
            flag_lists=flag_lists,
        )
        fig = pandas_plot(key, dfs, ndim)
        for ext in ("pdf", "png"):
            fig_fn = (cwd / f"{key}.{ext}").absolute()
            fig.suptitle(fig_fn)
            fig.tight_layout()
            fig.savefig(fig_fn)
        # fig = plo_plot_all_data(key, dfs)
        # ploplt.save_fig((cwd / f"{key}.html").absolute())


if __name__ == "__main__":
    run()
