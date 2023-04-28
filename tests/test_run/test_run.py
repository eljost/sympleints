from sympleints.main import run, parse_args


def test_run(this_dir):
    out_dir = (this_dir / "devel_ints").absolute()
    raw_args = f"--lmax 1 --lauxmax 1 --out-dir {out_dir}".split()
    args = parse_args(raw_args)
    run(args)
    for fn in out_dir.iterdir():
        fn.unlink()
    out_dir.rmdir()
