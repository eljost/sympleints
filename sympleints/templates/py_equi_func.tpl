def {{ equi_name }}({{ args }}, result):
    """See docstring of {{ name }}."""

    # Calculate values w/ swapped arguments
    {{ name }}({{ equi_args }}, result)
    {{ resort_func }}(result, {{ sizes|join(", ")}}, {{ ncomponents }})
