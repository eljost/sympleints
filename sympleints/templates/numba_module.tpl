"""
{{ header }}
"""

{% if comment %}
"""
{{ comment }}
"""
{% endif %}

import numba
from numba import i8, f8
import numpy
{% if boys %}
from pysisyphus.wavefunction.ints.boys import boys
{% endif %}

{% for ai in add_imports  %}
{{ ai }}
{% endfor %}

{% if resort_func == "resort_ba_ab" %}
@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
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

func_type = numba.types.FunctionType(
    {{ func_type }}
)

{% for func in funcs %}
{{ func.text }}
{% endfor %}

{{ driver_func }}
