def {{ equi_name }}({{ args }}, result):
    """See docstring of {{ name }}."""

    # Calculate values w/ swapped arguments
    {{ name }}({{ equi_args }}, result)
    # Swap two axes
    result[:] = numpy.moveaxis(result.reshape({{ reshape|join(",") }}), {{ from_axes }}, {{ to_axes }}).flatten()
