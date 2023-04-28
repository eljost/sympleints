"""
{{ header }}
"""

{{ comment }}

import numpy
{% if boys %}
from pysisyphus.wavefunction.ints.boys import boys
{% endif %}

{% for ai in add_imports  %}
{{ ai }}
{% endfor %}

{% for func in funcs %}
{{ func.text }}
{% endfor %}

{{ func_dict }}
