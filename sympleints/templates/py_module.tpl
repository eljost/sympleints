"""
{{ header }}
"""

{% if comment %}
"""
{{ comment }}
"""
{% endif %}

import numpy

{% if boys %}
# The default boys-import can be changed via the --boys-func argument.
# sympleints also includes a basic implementation in 'sympleints.boys'
from {{ boys }} import boys
{% endif %}

{% for ai in add_imports  %}
{{ ai }}
{% endfor %}

{% if resort_func == "resort_ba_ab" %}
def resort_ba_ab(result, sizea, sizeb, ncomponents):
    assert ncomponents > 0
    tmp = numpy.zeros_like(result)
    i = 0  # Original index
    sizeab = sizea * sizeb
    component_size = 0
    for _ in range(ncomponents):
        for b in range(sizeb):
            for a in range(sizea):
                j = component_size + (a * sizeb) + b  # New index
                tmp[j] = result[i]
                i += 1
        component_size += sizeab
    result[:] = tmp
{% endif %}

{% for func in funcs %}
{{ func.text }}
{% endfor %}

{{ func_dict }}
