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

func_type = numba.types.FunctionType(
    {{ func_type }}
)

{% for func in funcs %}
{{ func.text }}
{% endfor %}

{{ driver_func }}
