def {{ equi_name }}({{ args }}):
    """See docstring of {{ name }}."""

    # Swap two axes
    result = numpy.moveaxis({{ name }}({{ equi_args }}), {{ from_axes }}, {{ to_axes }})
    return result
