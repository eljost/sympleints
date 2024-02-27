@numba.jit(
    func_type.signature,
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def {{ equi_name }}({{ args }}, result):
    """See docstring of {{ name }}."""

    # Call equivalent function and write to result
    tmp = numpy.zeros_like(result)
    {{ name }}({{ equi_args }}, tmp)
    result[:]  {% if not primitive %}+{% endif %}= numpy.transpose(tmp.reshape({{ shape|join(", ") }}), axes={{ from_axes }}).flatten()

