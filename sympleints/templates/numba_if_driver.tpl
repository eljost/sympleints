{#
@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def {{ name }}({{ Ls|join(", ") }}, {{ args|join(", ") }}):
    {% for Lconds, func_name in conds_funcs %}
    {%+ if loop.index0 > 0 %}el{% endif %}if {{ Lconds|join(" and ") }}:
        {{ func_name }}({{ args|join(", ") }})
    {% endfor %}
    else:
        {{ args[-1] }}[:] = numpy.nan
#}

@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def {{ name }}({{ Ls|join(", ") }}, {{ args|join(", ") }}):
    {% for Lconds, func_name in conds_funcs %}
    {%+ if loop.index0 > 0 %}el{% endif %}if {{ Lconds|join(" and ") }}:
        func = {{ func_name }}
    {% endfor %}
    func({{ args|join(", ") }})
