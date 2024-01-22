import time

import numpy as np

from sympleints.boys import boys_quad, get_boys_func


def test_boys(atol=2e-10):
    nmax = 10
    boys = get_boys_func(nmax, order=6)

    ntaylor = 5000
    nlarge = 10
    tot_dur = 0.0
    quad_dur = 0.0
    for n in range(nmax + 1):
        xs = np.concatenate(
            (
                [0.0],
                np.random.rand(ntaylor) * 25,
                np.full(nlarge, 30.0) + 5 * np.random.rand(nlarge),
            )
        )
        for x in xs:
            dur = time.time()
            bc = boys(n, x)
            tot_dur += time.time() - dur
            dur = time.time()
            br = boys_quad(n, x)
            quad_dur += time.time() - dur
            assert abs(bc - br) <= atol, f"{n=}, {x=:.4f}, {bc=:.8e}, {br=:.8e}"
            # print(f"Checked {n=} at {x=:.8e}")
        # print()
    nevals = (nlarge + 1) * ntaylor
    per_eval = tot_dur / nevals
    per_quad_eval = quad_dur / nevals
    print(
        f"Eval duration: {tot_dur:.4f} s, {per_eval:.4e} s/eval, {per_quad_eval:.4e} s/quad eval"
    )
