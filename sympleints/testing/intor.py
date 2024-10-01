import numpy as np


def get_size_func(spherical):
    if spherical:

        def size_func(L):
            return 2 * L + 1

    else:

        def size_func(L):
            return (L + 2) * (L + 1) // 2

    return size_func


def intor_2c(
    shells,
    func_dict,
    ncomponents=1,
    spherical=False,
    **kwargs,
):
    """Basic driver for integrals over hermitian operators."""
    if spherical:
        nbfs = sum([shell.sph_size for shell in shells])
    else:
        nbfs = sum([shell.cart_size for shell in shells])

    L_max = max([shell.L for shell in shells])
    size_func = get_size_func(spherical)
    size_max = size_func(L_max)

    # Preallocate empty matrices and directly assign the calculated values
    integrals = np.zeros((ncomponents, nbfs, nbfs))

    a_start = 0
    result = np.empty(ncomponents * size_max**2)
    for i, shell_a in enumerate(shells):
        L_a = shell_a.L
        coeffs_a = shell_a.coeffs
        exps_a = shell_a.exps
        center_a = shell_a.center
        a_size = size_func(L_a)
        a_slice = slice(a_start, a_start + a_size)
        b_start = a_start
        for shell_b in shells[i:]:
            b_size = size_func(shell_b.L)
            batch_size = ncomponents * a_size * b_size
            b_slice = slice(b_start, b_start + b_size)
            func_dict[(L_a, shell_b.L)](
                exps_a[:, None],
                coeffs_a[:, None],
                center_a,
                shell_b.exps[None, :],
                shell_b.coeffs[None, :],
                shell_b.center,
                result=result[:batch_size],
                **kwargs,
            )
            integrals[:, a_slice, b_slice] = result[:batch_size].reshape(
                ncomponents, a_size, b_size
            )
            # Take symmetry into account
            if a_start != b_start:
                integrals[:, b_slice, a_slice] = np.transpose(
                    integrals[:, a_slice, b_slice], axes=(0, 2, 1)
                )
            b_start += b_size
        a_start += a_size

    # Return plain 2d array if components is set to 0, i.e., remove first axis.
    if ncomponents == 1:
        integrals = np.squeeze(integrals, axis=0)
    return integrals
