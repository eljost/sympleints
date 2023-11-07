@numba.jit(
    func_type.signature,
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def {{ equi_name }}({{ args }}, result):
    """See docstring of {{ name }}."""

    # np.moveaxis() is not yet supported by numba as of 0.58.1
    # result = numpy.moveaxis({{ name }}({{ equi_args }}), {{ from_axes }}, {{ to_axes }})

    # Call equivalent function and write to result
    {{ name }}({{ equi_args }}, result)
    result = numpy.transpose(result, axes={{ from_axes }})
