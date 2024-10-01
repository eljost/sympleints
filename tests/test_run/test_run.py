import tempfile

from sympleints.main import run, parse_args


def test_run():
    with tempfile.TemporaryDirectory(prefix="sympleints_test_") as out_dir:
        raw_args = f"--lmax 1 --lauxmax 1 --out-dir {out_dir}".split()
        args = parse_args(raw_args)
        run(l_max=args.lmax, l_aux_max=args.lauxmax)
